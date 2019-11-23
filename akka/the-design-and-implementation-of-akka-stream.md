# akka-stream(2.5.x)设计及实现

akka-stream构建于akka-actor之上并实现了[Reactive Streams](http://www.reactive-streams.org/)，不过这里主要介绍akka-stream的设计及实现细节，不会涉及到太多`Reactive Streams`的概念。

RunnableGraph从构建到运行大致可以分为3个阶段:

* build

  build阶段的主要目的是将inlet和outlet通过slot关联起来，最终获得一个traversal。traversal中所有的inlet和outlet都已经被连接。
  
* materialize

  materialize阶段的主要目的是根据traversal builder阶段获得的traversal分配执行单元(ActorGraphInterpreter)，一个RunnableGraph的async attribute决定最终会分配多少Actor。跨越async boundary的module通过Actor交互。
  
* running

  ActorGraphInterpreter运行阶段，直至upstream completed或者downstream completed或者出错。

从上层API来看materialize和running阶段的区分并不明显。当涉及到SubStreams时，SubStreams的build、materialize及running都是在parent-stream开始运行之后完成的。

## build

build相关的设计及代码都在`impl/package.scala`和`impl/TraversalBuilder.scala`文件中，akka-stream提供了一套DSL用于方便地构造任意形状的graph。build阶段有两个非常简单的stack language: materialize value和attribute，在build结束后会得到一个immutable graph blueprint，materialize阶段会根据这个blueprint计算最终的materialized value及根据attribute分配Actor。async(EnterIsland/ExitIsland) attribute会决定分配Actor的数量，换而言之，async attribute是决定stream吞吐量的主要参数之一。

下面介绍一下build阶段一些需要重点关注的地方:

### wire

每一个graph都有一个shape，shape表示拥有多少inlet和outlet，build阶段的一个主要目的就是通过slot将outlet和inlet连接在一起，相同slot的inlet和outlet被认为是连接在一起的。inlets根据添加的顺序被映射为连续的数字，比如：

```text
Module1[in1, in2, out] -> Module2[out] -> Module3[in]
```

每个inlet对应的slot为：

```text
Module1.in1 => 0
Module1.in2 => 1
Module3.in  => 2
```

给定inlet的slot之后，剩下的事情就是要将outlet映射为正确的slot，同时为了reusability，计算方式如下：

```text
slot = offsetOfModule + outToSlots(outlet.id)
```

其中`offsetOfModule`为module在整个graph的偏移量，对于上述示例，每个module的偏移量如下:

```text
Module1 => 0
Module2 => 2
Module3 => 2
```

换而言之，`offsetOfModule`为module的第一个inlet的slot，即便module可能没有inlets。

`outToSlots`为build阶段根据graph DSL计算得出的。

### reusability

```scala
val graph = GraphDSL.create() { implicit val builder =>
  import akka.stream.scaladsl.GraphDSL.Implicits._

  val flow = Flow.fromFunction[Int, Int](_ + 1)
  val source = Source.single(1)
  val sink = Sink.foreach(println)

  val f1 = builder.add(flow)
  val f2 = builder.add(flow)
  val f3 = builder.add(flow)

  source ~> f1 ~> f2 ~> f3 ~> sink
  ClosedShape
}

val system = ActorSystem("test")
val materializer = ActorMaterializer()(system)
RunnableGraph.fromGraph(graph).run(materializer)

system.terminate()

```

`reusability`是指在一个graph中一个module可能会出现多次，如上述例子所示，builder多次add同一个module(flow)，为了区分端口，akka-stream使用了port mapping。

当所有的端口都被连接时(即可以获得最终的Traversal)，每个重复的module会以不同的MaterializeAtomic出现在最终的Traversal中。

因为port mapping只是用来区分多次add同一个module的端口，所以在自己去实现GraphStage时需要把所有有状态变化的部分放到createStageLogic中。如果将可变的状态放到GraphStage的constructor中，则可能多个实例会共享可变状态，会导致undefined behavior。

### LinearTraversalBuilder

`LinearTraversalBuilder`允许最多有一个unwired outlet，因此可以将`LinearTraversalBuilder`分为两种情况：

* 只有一个outlet，如Flow、Source、Merge等
* 有多个outlet，但其中只有一个是unwired outlet，如Broadcast、FanOut等

`LinearTraversalBuilder`中的unwired outlet的offset为`-1`，即`outToSlots(unwiredOutlet) = -1`, 当在第一种情况下时默认条件成立，当在第二种情况下时默认情况不一定成立，需要通过`rewireLastOutTo`调整slot。

举一个简单的例子:

```scala
val graph = Source
  .single(1)
  .via(Flow.fromFunction(_.toString))
  .to(Sink.foreach(println))
```

以DSL的视角来看，构造graph的顺序为：

```text
Source -> Flow -> Sink
```

而在`LinearTraversalBuilder`的构造过程中，顺序和DSL顺序相反：

```text
Sink -> Flow -> Source
```

又因为`LinerTraversalBuilder`最多只有一个unwired outlet，则此unwired outlet所需要连接的inlet对应的slot必定为`offsetOfModule(unwiredOutlet) - 1`，所以默认将offset置为-1是正确的。

当为第二种情况时，由于unwired outlet连接的不一定为前一个module的最后一个inlet，所以需要通过`rewireLastOutTo`调整slot。具体细节可以参考`LinearTraversalBuilder`的实现。

### avoid forward wiring

为了能够仅遍历一次traversal就能够运行stream，在materialize阶段如果对于outlet没有找到对应的inlet，则会构造一个`ForwardWire`来记录当前outlet需要连接一个未知的inlet。当遍历到相应的inlet时则会通过`ForwardWire`连接outlet。在builder阶段可以通过改变module的添加顺序避免多余的`ForwardWire`，这样能够减少materialize的内存分配并使得materialize更快，例如：

```scala
val graph = GraphDSL.create() { implicit builder => 
  import akka.stream.scaladsl.GraphDSL.Implicits._

  val source1 = Source.single(1)
  val source2 = Source.single(2)
  val sink = Sink.foreach(println)
  val merge = Merge[Int](2)

  val mergeShape = builder.add(merge)

  source1 ~> mergeShape.in(0)
  source2 ~> mergeShape.in(1)
  mergeShape.out ~> sink

  ClosedShape
}

val system = ActorSystem("test")
val materializer = ActorMaterializer()(system)
RunnableGraph.fromGraph(graph).run(materializer)

system.terminate()
```

上述示例则会造成额外的`ForwardWire`，因为sink是在merge之后加入进去，则在meterialize阶段，merge.out需要连接到sink.in，但在meterialize merge时sink.in是未知的，所以需要一个额外的`ForwardWire`来记录连接信息。当改成如下所示代码时：

```scala
val graph = GraphDSL.create() { implicit builder =>
  import akka.stream.scaladsl.GraphDSL.Implicits._

  val source1 = Source.single(1)
  val source2 = Source.single(2)
  val sink = Sink.foreach(println)
  val merge = Merge[Int](2)

  val sinkShape = builder.add(sink)
  val mergeShape = builder.add(merge)

  source1 ~> mergeShape.in(0)
  source2 ~> mergeShape.in(1)
  mergeShape.out ~> sinkShape.in

  ClosedShape
}

val system = ActorSystem("test")
val materializer = ActorMaterializer()(system)
RunnableGraph.fromGraph(graph).run(materializer)

system.terminate()
```

因为sink是在merge之前加入的，在meterialize merge时能够找到sink.in，因此不需要保存额外的`ForwardWire`。

**NOTE**: 感兴趣的同学可以将`PhasedFusingActorMaterializer.Debug`打开，看看上述两个示例是否有相应的`ForwardWire`出现。

## materialize

materialize阶段根据build阶段计算出的`outToSlots`和attribute(EnterIsland/ExitIsland)将inlet和outlet连接，构造`ActorGraphInterpreter`用于运行`GraphStageLogic`。 一个graph有多少个Island就会有多少个Actor并行执行。

对于在同一个Island内连接的inlet和outlet，相关的logic都会在同一个Actor内执行。对于跨Island连接的inlet和outlet，数据则会在Actor之间传递。具体细节可以参考`BatchingActorInputBoundary`和`ActorOutputBoundary`的实现。

## running

一个`ActorGraphInterpreter`负责执行`GraphIsland`的逻辑。但由于fusing的缘故，一个`ActorGraphInterpreter`可能会执行多个stream的逻辑，比如sub-streams通过registerShell将`GraphInterpreterShell`注册到`ActorGraphInterpreter`。
`ActorGraphInterpreter`负责从`GraphIsland`或actor(`StageActor`)或`AsyncCallback`接受异步消息`BoundaryEvent`并委托给`GraphInterpreterShell`处理。

graph在materialize之后会立即执行，首先初始化`GraphStageLigic`，GraphInterpreter会根据每个Connection的状态及当前队列里的事件调用InHandler和OutHandler。

在running阶段，chased optimization避免了入队和出队的开销。

### chased optimization

chased optimization的目标是优化处理连续的pull/push，不需要将连续的pull/push事件入队，以上述reusability的代码为例:

在materialize开始后，sink在`preStart()`中会调用pull(in)，pull(in)会引发upstream的f3/f2/f1依次调用pull(in)，这一连串的pull(in)事件不会入队，而是直接在chased optimization阶段直接处理。

同理，当source.out的onPull被出发时，会通过push将数据发送到downstream并引发downstream的f1/f2/f3依次调用push，这一连串的push同样会在chased optimization阶段直接处理。

akka-stream的代码相对akka-actor较为复杂，上述仅仅是粗略的介绍了akka-stream的设计和实现，如果对实现细节感兴趣可以自行阅读代码。下面介绍akka-stream一些特性的实现细节。

## StageActor

为了将数据通过actor发送到stream，Source提供了actorRef:

```scala
def actorRef[T](
  completionMatcher: PartialFunction[Any, CompletionStrategy],
  failureMatcher: PartialFunction[Any, Throwable],
  bufferSize: Int,
  overflowStrategy: OverflowStrategy): Source[T, ActorRef] = {
    require(bufferSize >= 0, "bufferSize must be greater than or equal to 0")
    require(!overflowStrategy.isBackpressure, "Backpressure overflowStrategy not supported")
    Source
      .fromGraph(new ActorRefSource(bufferSize, overflowStrategy, completionMatcher, failureMatcher))
      .withAttributes(DefaultAttributes.actorRefSource)
  }
```

materialize结束后会得到一个Actor，所有发送到此Actor的消息都会发送到downstream。

actorRef的实现使用了`getAsyncCallback`，作为一种异步重入当前GraphStageLogic的手段，`getAsyncCallback`广泛应用于各种特性的实现，如akka-stream内部的KillSwitch、SubStreams及akka-remote-artery中的InboundControlJunction/OutboundControlJunction等。

### getAsyncCallback

getAsyncCallback返回一个AsyncCallback，通过AsyncCallback可以异步重入当前的GraphStageLogic(最终获取的callback为path-dependent type)，即在其他线程通过调用AsyncCallback.invoke(event)，最终的handler将会在GraphInterpreter环境下执行并正确的修改GraphStageLogic的状态。

在执行`callback.invoke(event)`时也做了一些优化措施:

```scala
(logic, event, promise, handler) => {
  val asyncInput = AsyncInput(this, logic, event, promise, handler)
  val currentInterpreter = GraphInterpreter.currentInterpreterOrNull
  if (currentInterpreter == null || (currentInterpreter.context ne self))
    self ! asyncInput
  else enqueueToShortCircuit(asyncInput)
}
```

`GraphInterpreter.currentInterpreterOrNull`会获取当前运行的`GraphInterpreter`，如果当前Actor正在运行，则直接入队`shortCircuit`，避免额外的Actor消息发送的开销。由于fusing的原因，这里的优化对于SubStreams尤为重要。

**NOTE**: 当多个线程同时调用AsyncCallback.invoke(event)时可能会造成race condition，即GraphInterpreter.currentInterpreter恰好是AsyncCallback所在的GraphStageLogic时，多个线程可能会同时调用interpreter.enqueueToShortCircuit，而enqueueToShortCircuit不是thread-safe的。在akka-stream的文档中也[提及了这一点](https://doc.akka.io/docs/akka/current/stream/stream-customize.html#using-asynchronous-side-channels)。

## SubStreams

顾名思义，SubStreams是指嵌入在stream中的stream。以`FlattenMerge: Graph[FlowShape[Graph[SourceShape[T], M], T], M]`为例，从类型不难看出，`FlatternMerge`有一个inlet和一个outlet，并从inlet接收`Graph[SourceShape[T], M]`，并将source中的数据通过outlet发送出去。

当upstream有新的source到来时，`FlattenMerge`执行如下代码:

```scala
def addSource(source: Graph[SourceShape[T], M]): Unit = {
  // If it's a SingleSource or wrapped such we can push the element directly instead of materializing it.
  // Have to use AnyRef because of OptionVal null value.
  TraversalBuilder.getSingleSource(source.asInstanceOf[Graph[SourceShape[AnyRef], M]]) match {
    case OptionVal.Some(single) => // optimization for single source
    case _ =>
      val sinkIn = new SubSinkInlet[T]("FlattenMergeSink")
      sinkIn.setHandler(new InHandler {
        override def onPush(): Unit = {
          if (isAvailable(out)) {
            push(out, sinkIn.grab())
            sinkIn.pull()
          } else {
            queue.enqueue(sinkIn)
          }
        }
        override def onUpstreamFinish(): Unit = if (!sinkIn.isAvailable) removeSource(sinkIn)
      })
      sinkIn.pull()
      sources += sinkIn
      val graph = Source.fromGraph(source).to(sinkIn.sink)
      interpreter.subFusingMaterializer.materialize(graph, defaultAttributes = enclosingAttributes)
  }
}
```

可以看出，substream的build、materialize和running阶段都是在`FlattenMerge`开始运行之后完成的，这里值得注意的一点是substream使用的materializer是`SubFusingMaterializerImpl`:

```scala
class SubFusingActorMaterializerImpl(
    val delegate: ExtendedActorMaterializer,
    registerShell: GraphInterpreterShell => ActorRef)
```

`SubFusingActorMaterializerImpl`会将`GraphInterpreterShell`通过registerShell注册到`ActorGraphInterpreter`中，因此substream及`FlattenMerge`的所有操作都是在同一个Actor中运行。

而从substream pull数据则是通过`AsyncCallback`做到的(代码在`SubSink`)，在上述getAsyncCallback部分也讲到了，pull event会直接入队`shortCircuit`而不是发送到Actor的mailbox。

## async

`Graph`的定义如下(省略了不相关的部分):

```scala
trait Graph[+S <: Shape, +M] {
  def async: Graph[S, M] = addAttributes(Attributes.asyncBoundary)

  def async(dispatcher: String) =
    addAttributes(Attributes.asyncBoundary and ActorAttributes.dispatcher(dispatcher))

  def async(dispatcher: String, inputBufferSize: Int) =
    addAttributes(
      Attributes.asyncBoundary and ActorAttributes.dispatcher(dispatcher)
      and Attributes.inputBuffer(inputBufferSize, inputBufferSize))
}
```

通过async可以指定Actor由哪个`Dispatcher`来执行，当调用async时，在builder阶段会将当前Graph包裹为一个独立的Island，在materialize阶段回为这个Island单独生成一个Actor。

但不同Island之间的inlet和outlet无法直接通过Connection连接，跨Island之间的inlet和outlet则通过`BatchingActorInputBoundary`和`ActorOutputBoundary`以Publisher/Subscriber的方式传输数据。

downstream可以通过添加async并行处理从upstream接收到的数据。

## error handling

akka-stream doc介绍了[几种](https://doc.akka.io/docs/akka/current/stream/stream-error.html)错误处理方式，这里主要介绍一下supervision，和akka-actor的supervision不同指出在于，如果想要使用akka-stream的supervision，则需要自己在`GraphStageLogic`中手动处理:

```scala
def createLogic(inheritedAttributes: Attributes): GraphStageLogic =
  new GraphStageLogic(shape) with InHandler with OutHandler {
    private def decider =
      inheritedAttributes.mandatoryAttribute[SupervisionStrategy].decider

    override def onPush(): Unit = {
      try {
        push(out, f(grab(in)))
      } catch {
        case NonFatal(ex) =>
          decider(ex) match {
            case Supervision.Stop => failStage(ex)
            case _                => pull(in)
          }
      }
    }

    override def onPull(): Unit = pull(in)

    setHandlers(in, out, this)
  }
```

上述为`Map`operator的实现，使用时可以根据需求定义decider:

```scala
val decider: Supervision.Decider = {
  case _: ArithmeticException => Supervision.Resume
  case _                      => Supervision.Stop
}
val flow = Flow[Int]
  .filter(100 / _ < 50)
  .map(elem => 100 / (5 - elem))
  .withAttributes(ActorAttributes.supervisionStrategy(decider))
val source = Source(0 to 5).via(flow)
```

## StreamRef

**TODO**: `StreamRef`是在akka 2.5.10之后加入的，内部的实现细节还未来得及看。

## integrating with akka-actor

关于如何将akka-stream和akka-actor组合在一起使用，Colin Breck写了一系列的文章:

[Integrating Akka Streams and Akka Actors: Part I](https://blog.colinbreck.com/integrating-akka-streams-and-akka-actors-part-i/)

[Integrating Akka Streams and Akka Actors: Part II](https://blog.colinbreck.com/integrating-akka-streams-and-akka-actors-part-ii/)

[Integrating Akka Streams and Akka Actors: Part III](https://blog.colinbreck.com/integrating-akka-streams-and-akka-actors-part-iii/)

[Integrating Akka Streams and Akka Actors: Part IV](https://blog.colinbreck.com/integrating-akka-streams-and-akka-actors-part-iv/)

## 其他

[这里](http://akarnokd.blogspot.com/2015/10/comparison-of-reactive-streams.html)是RxJava的核心开发者对不同`Reactive Streams`的实现做的性能对比，但由于作者对akka-stream的实现了解的不太多，因此对akka-stream的性能测试有很大偏差，并且这篇文章已经被翻译成[中文](https://blog.piasy.com/AdvancedRxJava/2017/03/19/comparison-of-reactive-streams-part-1/index.html)。
后来这位作者又去akka上开了个[issue](https://github.com/akka/akka/issues/20429)，问题在于这位作者所写的测试materialize会造成很大影响。akka内部有很多关于akka-stream的benchmark，有兴趣的同学可以仔细看一看。

后续有时间我会继续介绍akka如何基于akka-stream实现其他的网络协议。
