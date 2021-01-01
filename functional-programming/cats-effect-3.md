## cats effect 3

branch: series/3.x

commit: d144c74dd997b782e25227e9d1cff6ba0771f6aa

前段时间看了cats effect 3(CE3)的一些实现，这个[issue](https://github.com/typelevel/cats-effect/issues/634)详细描述了CE3在设计上的一些考量，其中包括typeclass继承关系以及一些语义上的改变。下面对CE3中的一些实现做一个简单的记录。

### runtime

CE3中runtime的实现为`IORuntime`:

```scala
final class IORuntime private[effect] (
  val compute: ExecutionContext,
  private[effect] val blocking: ExecutionContext,
  val scheduler: Scheduler,
  val config: IORuntimeConfig
)
final case class IORuntimeConfig private (
  val cancellationCheckThreshold: Int,
  val autoYieldThreshold: Int
)
```

其中:

* compute  跟cpu核数相等(或x2)的work-stealing线程池，用于调度fiber，默认为work-stealing pool
* blocking  数量不限(受物理资源限制)的线程池，用于执行阻塞的操作
* scheduler 用于执行定时任务，当前CE3中默认为一个线程
* config 
  * `cancellationCheckThreshold`表示在IOFiber执行多少次loop时去check是否被标记为取消
  * `autoYieldThreadshold`表示IOFiber在执行多少次loop时主动切出去

跟CE2中的主要区别为:

1. 默认的fiber scheduler由fixed thread pool变成了work-stealing thread pool，这也是CE3相对于cats effect 2(CE2)最重要的改进之一。关于CE3中work-stealing scheduler的设计可以参考[这篇文档](https://github.com/typelevel/cats-effect/blob/series/3.x/docs/schedulers.md)和这个[issue](https://github.com/typelevel/cats-effect/issues/1025)，主要参考了rust [tokio](https://tokio.rs/) scheduler的[设计](https://tokio.rs/blog/2019-10-scheduler#the-next-generation-tokio-scheduler)。感兴趣的同学可以去阅读CE3中的[实现](https://github.com/typelevel/cats-effect/blob/series%2F3.x/core/jvm/src/main/scala/cats/effect/unsafe/WorkStealingThreadPool.scala)，代码非常精简。其中tokio的blog中也提到了一个很有意思的观点:

   ``` 
   However, in practice the overhead needed to correctly avoid locks is greater than just using a mutex.
   ```

   即，有时候lock-free算法为了避免使用锁带来的开销比锁的开销还要大，比如CAS失败时可能需要重新执行之前的逻辑。

2. 增加了两个参数`cancellationCheckThreshold`和`autoYieldThreadshold`

   `autoYieldThreadshold`提供了fiber auto yield的功能，其中关于auto yield可以参考这里的[讨论](https://github.com/typelevel/cats-effect/issues/1126)。

   `cancellationCheckThreshold`相当于CE2 IORunLoop中的`maxAutoCancelableBatchSize`，在CE3中作为一个参数暴露给了使用者

在cats effect中，`Fiber`为最小的调度单元，关于`Fiber`的介绍可以参考[这篇文章](https://gist.github.com/djspiewak/d9930891d419c26fac1d58b5274f45ba)。其中提到:

```
This is an incredibly potent idea: "blocking" at a higher level of abstraction is nothing more than "descheduling" at a lower level. We repeat this trick once again within user space to create fibers.
```

即，我们在上层观测到的阻塞在底层都已经被调度出去了。以我们熟悉的多线程模型为例，当线程调用阻塞的操作，操作系统会将其调度出去；同样在用户层，`Fiber`如果调用阻塞的操作，调度器也应该将其调度出去。在CE3 IO Monad的实现中提供了`Blocking`，即当fiber执行`IO.blocking(blockingOps())`时，会将当前fiber调度到上面所说的blocking thread pool去执行，当blockingOps执行完成后会再切回到work-stealing pool。具体的实现如下:

```scala
case 20 =>
  val cur = cur0.asInstanceOf[Blocking[Any]]
  /* we know we're on the JVM here */
  if (cur.hint eq TypeBlocking) {
    resumeTag = BlockingR
    objectState.push(cur)
    runtime.blocking.execute(this)
  } else {
    runLoop(interruptibleImpl(cur, runtime.blocking), nextIteration)
  }

```

注意这里在切到blocking thread pool之前，将`resumeTag`置为`BlockingR`，然后在blocking thread pool中执行:

```scala
private[this] def blockingR(): Unit = {
  var error: Throwable = null
  val cur = objectState.pop().asInstanceOf[Blocking[Any]]
  val r =
    try cur.thunk()
    catch {
      case NonFatal(t) => error = t
    }

  if (error == null) {
    resumeTag = AfterBlockingSuccessfulR
    objectState.push(r.asInstanceOf[Object])
  } else {
    resumeTag = AfterBlockingFailedR
    objectState.push(error)
  }
  currentCtx.execute(this)
}
```

当执行完成后，通过`currentCtx.execute(this)`将fiber切回到compute thread pool。

因为fiber就是对顺序操作的抽象，因此在`IOFiber`的实现中记录了当前的continuation，每次切回后(包括async，blocking或auto yield)都会继续沿着continuation执行，感兴趣的同学可以去看看具体的实现。

CE3中`GenSpawn`提供了启动`Fiber`的功能:

```scala
trait GenSpawn[F[_], E] {
  def start[A](fa: F[A]): Fiber[F, E, A]
  def background[A](fa: F[A]): Resource[F, F[Outcome[F, E, A]]] = 
    Resource.make(fa.start)(_.cancel)(this).map(_.join)
}
```

* `start`用于启动一个新的`Fiber`，CE3中IO Monad的实现如下:

  ```scala
  case 14 =>
    val cur = cur0.asInstanceOf[Start[Any]]
    val initMask2 = childMask
    val ec = currentCtx
    val fiber = new IOFiber[Any](
      initMask2,
      null,
      cur.ioa,
      ec,
      runtime
    )
    rescheduleAndNotify(ec)(fiber)
    runLoop(succeeded(fiber, 0), nextIteration)
  ```

  可以看到，每当我们启动一个新的`Fiber`都会调用`rescheduleAndNotify`，`rescheduleAndNotify`会直接将`Fiber`入队到当前的`WorkerThread`的队列中，并唤醒挂起的线程(如果有的话，用于做fiber stealing)。

* `background`将启动的`Fiber`的生命周期嵌入到当前`Fiber`的生命周期，一个典型的使用场景就是，当收到请求后需要启动一个child fiber去执行一个任务，当parent fiber被取消时child fiber也能被取消，比如:

  ```scala
  def onRequest[F[_], A, B, C](param: A)(implicit F: GenSpawn[F]): F[(B, C)] = {
    val childTask: A => F[B] = // child task
    val defaultB: F[B] = // default if error or canceled
    val parentTask: A => F[C] = // parent task
    F.background(childTask(param)).use { joinToken =>
      for {
        c <- parentTask(param)
        b <- joinToken.flatMap(_ match {
          case Outcome.Succeeded(result) => result
          case _ => defaultB
        })
      } yield (b, c)
    }
  }
  ```

有时候可能会需要启动多个fiber来执行任务，那么可能就会用到`Supervisor`:

```scala
trait Supervisor[F[_]] {
  def supervise[A](fa: F[A]): F[Fiber[F, Throwable, A]]
}

object Supervisor {
  private class Token

  def apply[F[_]](implicit F: Concurrent[F]): Resource[F, Supervisor[F]] = {
    for {
      stateRef <- Resource.make(F.ref[Map[Token, F[Unit]]](Map())) { state =>
        state
          .get
          .flatMap { fibers =>
            // run all the finalizers
            fibers.values.toList.parSequence
          }
          .void
      }
    } yield {
      new Supervisor[F] {
        override def supervise[A](fa: F[A]): F[Fiber[F, Throwable, A]] =
          F.uncancelable { _ =>
            val token = new Token
            val action = fa.guarantee(stateRef.update(_ - token))
            F.start(action).flatMap { fiber =>
              stateRef.update(_ + (token -> fiber.cancel)).as(fiber)
            }
          }
      }
    }
  }
}
```

可以看到，`Supervisor`的实现很简单，通过一个`Map`来去管理所有的child fiber。一个简单的例子:

```scala
val task: F[Unit] = Supervisor[F].use { s =>
  val task1: F[Unit] = // task1
  val task2: F[Unit] = // task2
  val task3: F[Unit] = // task3
  for {
    fiber1 <- s.supervise(task1)
    fiber2 <- s.supervise(task2)
    fiber3 <- s.supervise(task3)
    _      <- fiber1.join >> fiber2.join >> fiber3.join
  } yield ()
}
```

那么，task如果被取消，则runtime保证task1, task2和task3**最终**都会被取消。下面来看一下cancelable和uncancelable task。

### MonadCancel

`MonadCancel` typeclass提供了cancellation，masking和finalization的功能，相当于CE2中的`Bracket`和部分`Concurrent`功能的集合。但相对于CE2，`MonadCancel`能够更加精确的描述cancelable和uncancelable的边界，具体如何使用可以参考[这篇文档](https://github.com/typelevel/cats-effect/blob/series%2F3.x/docs/typeclasses.md)。

在CE3 IO Monad的实现中，使用`masks: Int`来标记当前的取消状态，创建fiber都会指定一个初始的`initMask`(参见上面`start`)，uncancelable和unmask实现如下所示:

```scala
case 9 =>
  val cur = cur0.asInstanceOf[Uncancelable[Any]]

  masks += 1
  val id = masks
  val poll = new Poll[IO] {
    def apply[B](ioa: IO[B]) = IO.Uncancelable.UnmaskRunLoop(ioa, id)
  }
  conts.push(UncancelableK)
  runLoop(cur.body(poll), nextIteration)

case 10 =>
  val cur = cur0.asInstanceOf[Uncancelable.UnmaskRunLoop[Any]]

  if (masks == cur.id) {
    masks -= 1
    conts.push(UnmaskK)
  }
  runLoop(cur.ioa, nextIteration)
```

可以看到，每次调用`IO.uncancelable`都会将当前的masks加一，然后将`UncancelableK`推入continuations栈中，`poll`中记录最新的masks；unmask时，会判断当前masks是否匹配。`isUnmasked`用来判断当前是否可以取消:

```scala
private[this] def isUnmasked(): Boolean = masks == initMask
```

再具体使用时需要注意嵌套的`uncancelable`，比如对于一个`fa: F[Unit]`，下面三种写法语义上是相同的:

```scala
IO.uncancelable { outer =>
  IO.uncancelable { _ =>
    outer(fa)
  }
}

IO.uncancelable { _ =>
  IO.uncancelable { inner =>
    inner(fa)
  }
}

IO.uncancelable { _ => fa }
```

同时调用`fiber.cancel`时，也会检查是否当前处于unmasked状态，否则会一直等到unmasked才会中断fiber执行。

### Async

在CE3中，`Async`定义如下:

```scala
trait Async[F[_]] extends AsyncPlatform[F] with Sync[F] with Temporal[F] {
  def async[A](k: (Either[Throwable, A] => Unit) => F[Option[F[Unit]]]): F[A]
  def async_[A](k: (Either[Throwable, A] => Unit) => Unit): F[A] =
    async[A](cb => as(delay(k(cb)), None))
}
```

这里的`async`的参数类型为`(Either[Throwable, A] => Unit) => F[Option[F[Unit]]]`，前半部分`Either[Throwable, A] => Unit`是异步操作completed callback，在CE3 IO Monad的实现中是由IO runtime提供的，调用这个操作表明异步操作已完成，通知runtime继续执行fiber的continuations；后半部分`F[Option[F[Unit]]]`表示一个可选的当异步操作被取消时的清理工作。

CE2中的`async`就是这里的`async_`，同时在CE2中，如果`async`涉及到与第三方库交互，可能会导致fiber的continuation都会在第三方库的thread pool中执行，因此在CE2中，我们通常会在async调用结束通过`ContextShift`切换执行上下文。CE3完全抛弃了`ContextShift`，当调用completed callback时会保证将上下文切换为cats effect的执行上下文，简单来说，CE3是通过一个`AtomicReference`表示当前异步操作的状态`ContStat`，初始化为空`ContInitial`，当异步操作未完成时，将当前fiber标记为suspended状态，然后切出去；当异步操作完成时，即completed callback被调用时，会保存异步操作执行结果，并从suspended状态恢复，然后通过`asyncContinue`重新将fiber调度到执行队列，等待下一次执行。具体的实现比较复杂，这里就不再贴代码了，感兴趣的同学可以自己阅读代码。

### Dispatcher

通常异步操作有两种，前面提到的`async`通常表示回调函数只需要执行一次，通常我们还会遇到回调函数可能需要执行多次的情况，在CE2中通常会使用`Effect[F].runAsync(fa).unsafeRunSync()`来完成，而这种做法会导致`fa`的执行会在第三方的thread pool中执行，而`fa`中可能还会包含`start`或`async`等操作，`Dispatcher`可以用来解决这个问题，在CE3中，`Dispatcher`定义如下:

```scala
trait Dispatcher[F[_]] extends DispatcherPlatform[F] {
  def unsafeToFutureCancelable[A](fa: F[A]): (Future[A], () => Future[Unit])

  def unsafeToFuture[A](fa: F[A]): Future[A] =
    unsafeToFutureCancelable(fa)._1

  def unsafeRunAndForget[A](fa: F[A]): Unit = {
    unsafeToFutureCancelable(fa)
    ()
  }
}

trait DispatcherPlatform[F[_]] { this: Dispatcher[F] =>
  def unsafeRunSync[A](fa: F[A]): A =
    unsafeRunTimed(fa, Duration.Inf)

  def unsafeRunTimed[A](fa: F[A], timeout: Duration): A = {
    val (fut, cancel) = unsafeToFutureCancelable(fa)
    try Await.result(fut, timeout)
    catch {
      case t: TimeoutException =>
        cancel()
        throw t
    }
  }
}
```

这里`Dispatcher`相当于CE2中的`Effect`，但所有的执行都会在cats effect的runtime中执行，感兴趣的同学可以看看`Dispatcher`的实现。

当前的`Dispatcher`实现可能会导致deadlock，具体可以参考这个[issue](https://github.com/typelevel/cats-effect/issues/1499)，简单来说就是当执行`Dispatcher[F].unsafeRunSync(fa)`时，由于`Dispatcher`会启动一个fiber来执行`fa`，而当前线程需要阻塞地等待`fa` fiber完成，因此当所有的线程都调用`unsafeRunSync`时就会造成死锁。

### 其他

cats effect fiber与loom virtual thread相关的讨论: https://github.com/typelevel/cats-effect/issues/1057

另外，CE3中还引入了一些纯函数式的并行数据结构: https://github.com/typelevel/cats-effect/tree/series/3.x/std/shared/src/main/scala/cats/effect/std