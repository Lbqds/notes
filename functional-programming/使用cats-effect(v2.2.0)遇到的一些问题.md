## async vs concurrent 

cats-effect提供了async，用于执行异步运算，定义如下:

```scala
def async[A](k: (Either[Throwable, A] => Unit) => Unit): IO[A] =
  Async { (_, cb) =>
    val cb2 = Callback.asyncIdempotent(null, cb)
    try k(cb2)
    catch { case NonFatal(t) => cb2(Left(t)) }
  }
```

可以看到async的参数为一个callback，当异步操作完成时调用这个回调，乍一看好像很熟悉，其实async(异步操作)就相当于continuation。

对于concurrent，cats-effect提供了fiber的抽象，通过调用start来启动一个fiber，同样可以通过join等待fiber完成(跟thread类似)。

比如，对于下面的代码:

```scala
IO.async(cb => /* do something */).flatMap(x => /* continuation */)
```

不管如何，continuation都会在前面的async的回调执行完成后再执行，async只是不阻塞当前执行线程，所以有些时候我们需要考虑清楚是需要
async还是concurrent。

## shift

shift的作用是将特定的运算放入特定的ExecutionContext中执行，当我们与老的代码库交互时，可能会用到async，即异步操作在老的代码库的专属线程池里运行，
而cats-effect当前的实现会导致异步操作的continuation也会在老的代码库中执行。为了避免这个问题，我们需要在执行完特定的异步任务之后通过调用shift
将运行上下文切回到当前的线程池。

具体示例请参考[这里](https://blog.softwaremill.com/thread-shifting-in-cats-effect-and-zio-9c184708067b)

## cancelable

cats-effect的cancelable提供了取消运算的功能，async接口相当于是构建了一个不可取消的异步运算。cats-effect IO在运行时只会在遇到async
操作时才会检测是否运算被取消，所以我们可以将cancelBoundary插入到运算中。

更多时候我们需要自己提供线程安全的取消逻辑，具体请参考cats-effect的文档: [Cancellation is a Concurrent Action](https://typelevel.org/cats-effect/datatypes/io.html#gotcha-cancellation-is-a-concurrent-action)


## tracing

cats-effect或者fs2的实现都是先构造出AST，然后使用解释器来解释运行，与interpreter design pattern类似。但这种做法有一个比较令人头疼的地方就是，
当需要debug时，所有的stack trace都是interpreter的stack trace，很难看出问题出在哪，好在cats-effect在v2.2.0中加入了tracing，通过在构造AST时
加入class信息，具体使用方式可以参考cats-effect [tracing文档](https://typelevel.org/cats-effect/tracing).
