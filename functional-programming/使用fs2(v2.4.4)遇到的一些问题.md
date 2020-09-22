## fs2的一些实现细节

fs2使用Free Monad(FreeC)构造stream，并通过解释器运行stream。FreeC(Algebra.scala)会捕获异常和中断，分别对应ADT `Fail`和`Interrupted`.
fs2 stream构造会通过trampoline来保证stack safety。Algebra.scala中实现的compile方法则用于解释执行stream。

在stream运行过程中可能会涉及到资源的获取和清理，如打开关闭连接或打开关闭文件等，为了保证资源在stream终止时(异常，中断或stream执行完成)能够自动释放，
fs2引入了scope(Scope.scala)，每个stream开始执行时都会创建一个root scope，stream执行期间可能会创建多个scope，这些scope构成了一个树状结构，
每个scope都可能会有多个child scope，除root scope外的每个scope都会有一个parent scope. scope可以分为两种，可中断的和不可中断的，
分别通过scope和interruptScope来构造(这两个方法位于Algebra.scala)，scope有两个相关的ADT，`OpenScope`和`CloseScope`，在解释器执行时，
`OpenScope`会在当前scope下创建一个child scope，当stream运行时所有通过`Acquire`获取的资源都会注册到当前scope下; `CloseScope`会将清理
所有注册的资源，并关闭所有的child scope。当stream运行时如果中断发生(通过调用interruptWhen)或者计算抛出exception，也会清理所有资源并关闭所有child scope。

## infinite stream

在fs2官方文档中，大多都是finite stream，而在我们正常使用过程中，更多的可能是infinite stream，比如网络数据流等。比如在libp2p协议中，每个网络
连接都会通过muxer分为多个stream(类似于QUIC，为了与fs2 stream区分，下面都使用Libp2pStream)，因此每个连接都会有无数个Libp2pStream，而每个监听
端口可能会有无数个连接，同时每个节点又会监听多个端口，为了方便，我们使用下面的代码来模拟这种场景:

```scala
import scala.util.Random
final case class Connection(id: Int, endpoint: String) {
  override def toString: String = s"connection $id, endpoint: $endpoint"

  def libp2pStreams: fs2.Stream[IO, Libp2pStream] = {
    fs2.Stream.eval(IO.delay(Libp2pStream(this, Random.nextInt))) ++
    fs2.Stream.sleep_(2 seconds) ++
    fs2.Stream.suspend(libp2pStreams)
  }
}

final case class Libp2pStream(conn: Connection, streamId: Int) {
  override def toString: String = s"new libp2p stream $streamId from $conn"
}

def connectionStream(endpoint: String): fs2.Stream[IO, Connection] = {

  def start(num: Int): fs2.Stream[IO, Connection] = {
    fs2.Stream.eval(IO.pure(Connection(num, endpoint))) ++
    fs2.Stream.sleep_(5 seconds) ++
    fs2.Stream.suspend(start(num + 1))
  }

  start(0).evalTap(c => IO.delay(println(c)))
}

def endpoints: fs2.Stream[IO, String] = fs2.Stream("endpoint1", "endpoint2", "endpoint3")
```

上面的代码，我们定义了有三个监听地址(`endpoints`)，每个监听地址有一个infinite connection stream，通过connectionStream来构造，
每个connection有一个infinite Libp2pStream stream，对于每一个监听地址，每5s会有一个新的连接，每个连接每两秒会有一个新的Libp2pStream，
下面我们尝试去运行所有的Libp2pStream:

```scala
def runAllLibp2pStreams1: IO[Unit] = {
  endpoints
    .flatMap(e =>
      connectionStream(e)
        .flatMap(_
          .libp2pStreams
          .evalTap(s => IO.delay(println(s)))
        )
    )
    .compile
    .drain
}

def runAllLibp2pStreams2: IO[Unit] = {
  endpoints
    .map(e =>
      connectionStream(e)
        .flatMap(_
          .libp2pStreams
          .evalTap(s => IO.delay(println(s)))
        )
    )
    .parJoinUnbounded
    .compile
    .drain
}

def runAllLibp2pStreams3: IO[Unit] = {
  endpoints
    .flatMap(e =>
      connectionStream(e)
        .map(_
          .libp2pStreams
          .evalTap(s => IO.delay(println(s)))
        )
    )
    .parJoinUnbounded
    .compile
    .drain
}

def runAllLibp2pStreams4: IO[Unit] = {
  endpoints
    .map(e =>
      connectionStream(e)
        .map(_
          .libp2pStreams
          .evalTap(s => IO.delay(println(s)))
        )
    )
    .parJoinUnbounded
    .parJoinUnbounded
    .compile
    .drain
}
```

上面四个方法从类型层面来看好像都没什么问题，但结果完全不同，对于`runAllLibp2pStream1`，只会执行第一个endpoint的第一个connection；对于
`runAllLibp2pStream2`，会执行所有endpoint的第一个connection；对于`runAllLibp2pStream3`，只会执行第一个endpoint的所有connection，
对于`runAllLibp2pStream4`，会执行所有endpoint的所有connection。 主要问题就在于flatMap(或者说是Monad)只是对顺序计算的抽象，
而这里我们需要并行运算所有的connection的Libp2pStream.

对于这个例子，每一个连接对应的类型应该为: `fs2.Stream[F, Libp2pStream]`，而每个监听地址都可能会有多个连接，所以一个监听地址的Libp2pStream的类型应该为: `fs2.Stream[F, f2.Stream[F, Libp2pStream]]`，
而一个节点可能会有多个监听多个地址，所以所有监听地址的Libp2pStream的类型应该为: `fs2.Stream[F, fs2.Stream[F, fs2.Stream[F, Libp2pStream]]]`.
所以`runAllLibp2pStream4`才是我们想要的结果，感兴趣的同学可以看我在libp2p中的[实现](https://github.com/Lbqds/libp2p/blob/pubsub/src/main/scala/libp2p/host/Network.scala)

## interruptWhen和onFinalize

fs2提供了onFinalize和onFinalizeCase接口，用于当stream终止时根据终止原因执行一些清理工作。而interruptWhen可以异步的去终止stream，所以有时
我们可能会需要同时调用interruptWhen和onFinalize，即我们可能会异步的去中断stream，然后执行finalizer，对于下面这段代码:

```scala
import cats.effect.concurrent.Deferred
val stream = fs2.Stream.eval(IO(1) <* sleep(1 seconds)).repeat
Deferred[IO, Either[Throwable, Unit]].flatMap { promise =>
  stream
    .evalMap(value => IO(println(value)))
    .interruptWhen(promise)
    .onFinalizeCase {
      case ExitCase.Canceled => IO(println("canceled"))
      case ExitCase.Completed => IO(println("completed"))
      case ExitCase.Error(err) => IO(println(s"error $err"))
    }
    .compile
    .drain
    .start
    .void >>
    sleep(6 seconds) >> promise.complete(Right(())) >> sleep(2 seconds)
}
```

这段代码的输出并非我们期望的`canceled`，而是`completed`。如果我们调换interruptWhen跟onFinalizeCase的顺序:

```scala
import cats.effect.concurrent.Deferred
val stream = fs2.Stream.eval(IO(1) <* sleep(1 seconds)).repeat
Deferred[IO, Either[Throwable, Unit]].flatMap { promise =>
  stream
    .evalMap(value => IO(println(value)))
    .onFinalizeCase {
      case ExitCase.Canceled => IO(println("canceled"))
      case ExitCase.Completed => IO(println("completed"))
      case ExitCase.Error(err) => IO(println(s"error $err"))
    }
    .interruptWhen(promise)
    .compile
    .drain
    .start
    .void >>
    sleep(6 seconds) >> promise.complete(Right(())) >> sleep(2 seconds)
}
```

现在我们发现，调换顺序之后，输出的结果则为我们想要的`canceled`。出现这种情况的原因为，onFinalizeCase会引进一个scope，interruptWhen会引进
一个interrupt scope，而第一种情况，interruptWhen引进的interrupt scope会成为onFinalizeCase引进的scope的child scope，所以当中断发生
时，并不会将中断传播到parent scope；而第二种情况恰恰相反，interrupt scope为parent scope，当中断发生时，interrupt scope会关闭所有child scope，
所以会输出我们期望的结果。

除此之外interruptWhen也可以用于同时停止多个并行的stream。

## cats effect Resource

当我们使用cats-effect时，经常会使用到Resource，比如fs2-io提供的tcp/udp连接返回的均为Resource，比如对于server端监听结果返回的为:
`fs2.Stream[F, Resource[F, Connection]]`，对于cats-effect Resource，fs2提供了Stream.resource方法来操纵Resource，resource
方法会引入一个新的scope，在此scope终止之前都可以使用这个resource。也可以通过resourceWeak(即bracketWeak)来扩展resource的操作范围，具体的示例可以参考
[fs2-chat](https://github.com/typelevel/fs2-chat/blob/master/src/main/scala/fs2chat/server/Server.scala)。

## 关于如何更好地使用fs2

学习使用fs2的[开源项目](https://fs2.io/ecosystem.html)，如果遇到问题也可以去fs2 gitter中请求帮助。
