# Algebraic Effects and Handlers -- Part I

最近花时间简单了解了一下algebraic effects，作为函数式编程中较为前沿的概念，algebraic effects有很多实现(基本都是用于research)，有些是直接在语言层实现，比如[koka](https://github.com/koka-lang/koka)、[unison](https://github.com/unisonweb/unison)、[effekt](https://github.com/effekt-lang/effekt)等；有些是以库的形式提供，比如[scala-effekt](https://github.com/b-studios/scala-effekt)、[eff](https://github.com/atnos-org/eff)、[freer-effects](https://github.com/IxpertaSolutions/freer-effects)。对于不同的实现，我打算记录一下我自己学习的过程及想法，这是第一篇，用delimited continuation来实现algebraic effects。

## introduction

对于algebraic effects是什么及能做什么已经有一些不错的教程:

* [An Introduction to Algebraic Effects and Handlers](https://www.eff-lang.org/handlers-tutorial.pdf)
* [Concurrent Programming with Effect Handlers](https://github.com/ocamllabs/ocaml-effects-tutorial)
* [koka effect handlers](https://koka-lang.github.io/koka/doc/book.html#sec-handlers)

## delimited continuation

scala提供了first class delimited continuation，可以直接用来实现effect handlers，比如对于`Amb`:

```scala
trait Amb[T, R] {
  def flip(): Boolean @cpsParam[T, R]
}

def ambList[T]: Amb[T, List[T]] = new Amb[T, List[T]] {
  override def flip(): Boolean @cpsParam[T, List[T]] = shift { (k: Boolean => T) =>
    List(k(false)) ++ List(k(true))
  }
}

def handle[T, R](program: Amb[T, R] => T @cpsParam[T, R])(amb: Amb[T, R]): R = reset(program(amb))

handle[Int, List[Int]]{ amb =>
  val p = amb.flip()
  if (p) 1 else 2
}(ambList[Int])
```

上面直接通过delimited continuation来实现`Amb`，但当前scala delimited continuation还存在不少问题:

* 当前delimited continuation是社区维护的compiler plugin来实现的，但对于scala 2.13后的版本已经无人维护
* 当前的delimited continuation仍然存在一些bug，比如上述的代码如果直接写成`if (amb.flip()) 1 else 2`编译会出错(***可能***是在做ANF转换时出了问题)
* 编译错误提示晦涩难懂

当然除此之外***直接***用delimited continuation来实现effect handlers的最大问题在于很难组合不同的effect，主要原因在于`shift`只能捕获至最内层`reset`的delimited continuation(即`static delimited continuation`)，这一点后面还会提到。但通过delimited continuation来实现effect handlers的思路并没有什么太大的问题，因此先实现delimited continuation来解决上面几个问题，如果熟悉函数式编程的话，提到continuation可能首先想到的就是Monad。那么接下来就通过Monad来实现delimited continuation，`Control` monad及`shift/reset`定义如下：

```scala
sealed trait Control[A] {
  def map[B](f: A => B): Control[B] = flatMap((a: A) => Pure(f(a)))
  def flatMap[B](f: A => Control[B]): Control[B] = Bind(this, f(_))
  def andThen[B](c: Control[B]): Control[B] = Bind(this, (_: A) => c)
}

final case class Pure[A](value: A) extends Control[A]
final case class Delay[A](thunk: () => A) extends Control[A]
final case class Bind[A, B](fa: Control[A], f: A => Control[B]) extends Control[B]

def shift[A, B, C](f: (A => Control[B]) => Control[C]): Control[A]
def reset[B, C](b: Control[B]): Control[C]
```

* Delay: `thunk`表示effectful computation
* Bind: `fa`计算完成后将结果传入`f`，即`f`为`fa`的continuation
* Shift: 需要一个类型为`(A => Control[B]) => Control[C]`的参数`f`，其中`A => Control[B]`为`shift`捕获到的delimited continuation，`shift`的返回类型为`Control[A]`。

为了对这些类型参数有一个更直观的理解，下面举一个简单的示例：

```scala
def isEven(num: String): Control[Boolean] = reset {
  for {
    x <- shift { strToInt: (String => Control[Int]) =>
      strToInt(num).map(x => (x & 1) == 0)
    }
  } yield x.toInt
}
```

`isEven`判断输入的用字符串表示的数字是否是偶数(简单起见，忽略了错误处理)。这里`shift`捕获到的delimited continuation为`(x: String) => Pure(x.toInt)`，对应的类型为`String => Control[Int]`，最终返回的结果为`Control[Boolean]`，加上类型标注后的代码如下所示：

```scala
def isEven(num: String): Control[Boolean] = reset[Int, Boolean] {
  for {
    x <- shift[String, Int, Boolean] { strToInt: (String => Control[Int]) =>
      strToInt(num).map(x => (x & 1) == 0)
    }
  } yield x.toInt
}
```

在上面的定义中不难看出，`Bind(fa, f)`中`f`为`fa`的continuation，如果想要获取delimited continuation，那么一个最直接的做法就是在构造时在`f`的某一处插入一个标记，比如：

```scala
fa.flatMap(f1).flatMap(f2).flatMap(f3).delimited.flatMap(f4)
```

其中`f1 compose f2 compose f3`就是我们需要的delimited continuation，为了方便描述，还需要额外的两个数据类型：

```scala
final case class Cont[A, B, C](f: (A => Control[B]) => Control[C]) extends Control[A]
final case class Delimited[B, C](inner: Control[B]) extends Control[C]
```

接着定义一些辅助函数：

```
def apply[A](a: A): Control[A] = Pure(a)
def delay[A](a: => A): Control[A] = Delay(() => a)
def shift[A, B, C](f: (A => Control[B]) => Control[C]): Control[A] = Cont(f)
def reset[A, B](c: Control[A]): Control[B] = Delimited[A, B](c)
```

下面需要写一个的interpreter来运行`Control`：

```scala
type Func = Any => Control[Any]

def run[A](p: Control[A]): A = {

  def fold(cont: List[Func]): Func =
    cont.tail.foldLeft(cont.head) { case (acc, item) =>
      (a: Any) => acc(a).flatMap(item)
    }
  
  def loop(inner: Control[Any], cont: List[Func]): Any = inner match {
    case Pure(v) => if (cont.isEmpty) v else {
      loop(cont.head(v), cont.tail)
    }
    case Delay(thunk) => if (cont.isEmpty) thunk() else {
      loop(cont.head(thunk()), cont.tail)
    }
    case Bind(fa, f) => loop(fa, f.asInstanceOf[Func] +: cont)
    case Delimited(inner) => if (cont.isEmpty) loop(inner, List.empty) else {
      loop(cont.head(loop(inner, List.empty)), cont.tail)
    }
    case Cont(f) => loop(f(fold(cont)), List.empty)
  }
  
  loop(p.asInstanceOf[Control[Any]], List.empty).asInstanceOf[A]
}
```

由于`shift`捕获到的是到最内层的`reset`的delimited continuation，因此上面的代码在处理`Delimited`时，`loop(cont.head(loop(inner, List.empty)))`中内层的`loop`调用会先去执行`inner`。当遇到`Cont`时，`loop(f(fold(cont)), List.empty)`，其中`fold(cont)`就是`shift`捕获到的delimited continuation。

为了确保stack safety，可以将interpreter修改成如下代码：

```scala
def run[A](p: Control[A]): A = {

  def loop(inner: Control[Any]): Any = {
    val pending = ListBuffer.empty[(Func, List[Func])]
    
    def fold(cont: List[Func]): Func =
      cont.tail.foldLeft(cont.head) { case (acc, item) =>
        (a: Any) => acc(a).flatMap(item)
      }
    
    @tailrec
    def step(inner: Control[Any], cont: List[Func]): Any = {
      inner match {
        case Pure(v) => if (cont.isEmpty) v else {
          step(cont.head(v), cont.tail)
        }
        case Delay(thunk) => if (cont.isEmpty) thunk() else {
          step(cont.head(thunk()), cont.tail)
        }
        case Bind(fa, f) =>
          step(fa, f.asInstanceOf[Func] +: cont)
        case Delimited(inner) => if (cont.isEmpty) step(inner, List.empty) else {
          pending.prepend((cont.head, cont.tail))
          step(inner, List.empty)
        }
        case Cont(f) =>
          step(f(fold(cont)), List.empty)
      }
    }
    
    var result = step(inner, List.empty)
    while (pending.nonEmpty) {
      val (head, tail) = pending.remove(0)
      result = step(head(result), tail)
    }
    result
  }
  
  loop(p.asInstanceOf[Control[Any]]).asInstanceOf[A]
}
```

上述将非tail-recursive的代码转换为tail-recursive的代码，即将递归调用开始到函数末尾的计算(也是delimited continuation)存入栈中，最后再依次出栈执行相应的计算即可，这种通用的转换方式也可以单独抽出来，scala中的[Trampoline](http://blog.higher-order.com/assets/trampolines.pdf)做的就是这个事情，只不过Trampoline是monad(`type Trampoline[A] = Free[Function0, A]`)，通过flatMap来捕获continuation。

在上面的实现中，如果`shift`外层没有`reset`则表示捕获当前的continuation，如果`reset`内没有`shift`则表示不捕获continuation。最终的目标是实现effect handlers，因此为了避免误用，可以选择避免将`shift`和`reset`直接暴露出来。

下面回到之前提到的问题，即通过delimited continuation很难组合不同的effect handler，前面也说过主要原因在于`shift`只能捕获至最内层的`reset`的delimited continuation，举个例子:

```scala
trait Amb {
  def flip(): Control[Boolean]
}

trait State[T] {
  def set(v: T): Control[Unit]
  def get: Control[T]
}

trait Handler[R] {
  def apply(prog: this.type => Control[R]): Control[R] = handle(prog)
  def handle(prog: this.type => Control[R]): Control[R] = reset(p(this))
  def use[A](p: (A => Control[R]) => Control[R]): Control[A] = shift(p)
}

def ambList[T] = new Amb with Handler[List[T]] {
  override def flip(): Control[Boolean] = use[Boolean, List[T]] { resume =>
    for {
      l <- resume(true)
      r <- resume(false)
    } yield l ++ r
  }
}

def state[T, R](init: T) = new State[T] with Handler[R] {
  private var value: T = init
  override def set(v: T): Control[Unit] = use { resume =>
    Control.Delay(() => value = v).andThen(resume(()))
  }
}
```

上述代码定义了两个effect type(`Amb`和`State`)及其相应的handler(`ambList`和`state`)，接下来我们尝试组合这两个effect:

```scala
def program(s: State[Int], amb: Amb): Control[Int] = for {
  x <- s.get()
  b <- amb.flip()
  _ <- if (b) s.set(x + 1) else Control(())
  y <- s.get()
} yield x + y

val p = ambList[Int]((a: Amb) =>
  state(0)(s =>
    program(s, a).map(List.apply(_))
  )
)
println(run(p))
```

上述的代码可以正常运行并输出`List(1, 1)`，但如果我们改成如下所示的代码:

```scala
val p = ambList[Int]((a: Amb) =>
  state(0)(s =>
    program(s, a)
  ).map(List.apply(_))
)
println(run(p))
```

代码同样可以编译通过，但我们上面实现的解释器在执行时会抛出异常:

```
Exception in thread "main" java.lang.ClassCastException: java.lang.Integer cannot be cast to
scala.collection.immutable.List
```

原因就在于，`program(s, a)`外层的`reset`为state handler，因此在执行`flip`时只能捕获至state handler的delimited continuation:

```scala
(b: Boolean) => (if (b) s.set(x + 1) else Control(())).flatMap(_ => s.get()).map(y => x + y)
```

类型为`(b: Boolean) => Control[Int]`，而`ambList[Int]`的定义需要的是`(b: Boolean) => Control[List[Int]]`，因此在解释器执行时抛出了上述异常。

同样，解决这个问题最直接的方法就是，对于每个handler，在continuation中插入与之唯一对应的标记，然后在运行时捕获至标记处的delimited continuation即可。

下面我们修改上述的一些定义，如下所示:

```scala
sealed trait Control[A] {
  def map[B](f: A => B): Control[B] = flatMap((a: A) => Pure(f(a)))
  def flatMap[B](f: A => Control[B]): Control[B] = Bind(this, f(_))
  def andThen[B](c: Control[B]): Control[B] = Bind(this, (_: A) => c)
}

final case class Pure[A](value: A) extends Control[A]
final case class Delay[A](thunk: () => A) extends Control[A]
final case class Bind[A, B](fa: Control[A], f: A => Control[B]) extends Control[B]
final case class Cont[A, B, C](handler: Handler[C], f: (A => Control[B]) => Control[C]) extends Control[A]
final case class Prompt[A, B](handler: Handler[B], inner: Control[A]) extends Control[B]

trait Handler[Res] {
  def apply(p: this.type => Control[Res]): Control[Res] = handle(p)
  def handle(p: this.type => Control[Res]): Control[Res] = Prompt(this, p(this))
  def use[A](p: (A => Control[Res]) => Control[Res]): Control[A] = Cont(this, p)
}
```

这里我们直接使用handler作为标记，当遇到`Prompt`时在pending中插入对应的`Mark`，当遇到`Cont`时捕获从当前位置开始至对应的`Mark`处的delimited continuation。

```scala
type Func = Any => Control[Any]

private sealed trait Step
private final case class Computation(f: Func) extends Step
private final case class Mark(handler: Handler[_]) extends Step

private val controlId: Func = (x: Any) => Pure(x)

private final class Pending(var steps: List[Step] = List.empty) { self =>
  def cont(mark: Handler[_]): Func = {
    
    @tailrec
    def loop(acc: Func, steps: List[Step]): Func = steps match {
      case Nil => throw new RuntimeException("internal error")
      case head :: tail => head match {
        case Computation(f) =>
          val newAcc = (x: Any) => acc(x).flatMap(f)
          loop(newAcc, tail)
        case Mark(h) =>
          if (h == mark) {
            self.steps = steps
            acc
          } else {
            val newAcc = (x: Any) => Prompt(h, acc(x)).asInstanceOf[Control[Any]]
            loop(newAcc, tail)
          }
      }
    }
    
    loop(controlId, steps)
  }
  
  @tailrec
  def pop(): Option[Func] = steps match {
    case head :: tail => head match {
      case s: Computation => steps = tail; Some(s.f)
      case _ => steps = tail; pop()
    }
    case Nil => None
  }
  
  def push(step: Step): Unit = steps = step +: steps
}

def run[A](p: Control[A]): A = {
  val pending = new Pending
  
  @tailrec
  def loop(c: Control[Any]): Any = c match {
    case Pure(v) => pending.pop() match {
      case None => v
      case Some(f) => loop(f(v))
    }
    case Delay(thunk) => pending.pop() match {
      case None => thunk()
      case Some(f) => loop(f(thunk()))
    }
    case Bind(fa, f) =>
      pending.push(Computation(f.asInstanceOf[Func]))
      loop(fa)
    case Prompt(handler, inner) =>
      pending.push(Mark(handler))
      loop(inner)
    case Cont(handler, f) =>
      loop(f(pending.cont(handler)))
  }
  
  loop(p.asInstanceOf[Control[Any]]).asInstanceOf[A]
}
```

在[algebraic effects for functional programming](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/08/algeff-tr-2016-v2.pdf)这篇论文中ambiguity resumption部分给出了一个例子，即不同的handler顺序结果可能不同，比如:

```scala
val p1 = ambList[Int]((a: Amb) =>
  state(0)(s =>
    program(s, a).map(List.apply(_))
  )
)

val p2 = state(0)((s: State[Int]) =>
  ambList[Int]((a: Amb) =>
    amb(s, a).map(List.apply(_))
  )
)
```

在上述的实现中，并不会出现ambiguity。但可以借鉴[scala-effekt](https://github.com/b-studios/scala-effekt)的实现，提供一个`StateHandler`，插入标记时backup当前的state，然后在handler处理结束时restore state。即:

```scala
trait StateHandler[Res] extends Handler[Res] {

  def backup(): Unit = _backup = data
  def restore(): Unit = data = _backup

  private var data = Map.empty[Field[_], Any]
  private var _backup: Map[Field[_], Any] = _
  
  // from scala-effekt
  def Field[T](value: T): Field[T] = {
    val field = new Field[T]()
    data = data.updated(field, value)
    field
  }

  // all the field data is stored in `data`
  class Field[T] () {
    def value: Control[T] = Control(data(this).asInstanceOf[T])
    def value_=(value: T): Control[Unit] = Control.delay {
      data = data.updated(this, value)
    }
    def update(f: T => T): Control[Unit] = for {
      old <- value
      _   <- value_=(f(old))
    } yield ()
  }
}
```

首先，当插入标记时先去backup当前的状态，即`Prompt`如果handler为`StateHandler`时调用backup()：

```scala
def run[A](p: Control[A]): A = {
  val pending = new Pending
  @tailrec
  def loop(c: Control[Any]): Any = c match {
    case Pure(v) => pending.pop() match {
      case None => v
      case Some(f) => loop(f(v))
    }
    case Delay(thunk) => pending.pop() match {
      case None => thunk()
      case Some(f) => loop(f(thunk()))
    }
    case Bind(fa, f) =>
      pending.push(Computation(f.asInstanceOf[Func]))
      loop(fa)
    case Prompt(handler, inner) =>
      pending.push(Mark(handler))
      handler match {
        case h: StateHandler[_] => h.backup()
        case _ =>
      }
      loop(inner)
    case Cont(handler, f) =>
      loop(f(pending.cont(handler)))
  }
  loop(p.asInstanceOf[Control[Any]]).asInstanceOf[A]
}
```

同样在handler处理结束后，调用restore()恢复状态:

```scala
@tailrec
def pop(): Option[Func] = steps match {
  case head :: tail => head match {
    case s: Computation => steps = tail; Some(s.f)
    case Mark(handler) =>
      handler match {
        case h: StateHandler[_] => h.restore()
        case _ =>
      }
      steps = tail
      pop()
  }
  case Nil => None
}
```

最终的代码在[这里](https://github.com/lbqds/effect-handlers).

## scala-effekt

[scala-effekt](https://github.com/b-studios/scala-effekt)也是通过delimited continuation来实现effect handlers，大体思路和上述类似，都是来源于 *a monadic framework for delimited continuations* 这篇论文。但和上述的实现不同，scala-effekt将continuation使用单独的数据类型`MetaCont`来表示，因此interpreter的实现也不同。除此之外，为了支持ambiguity resumption，scala-effekt提供了`State` handler。

## reference

1. [An Introduction to Algebraic Effects and Handlers](https://www.eff-lang.org/handlers-tutorial.pdf)
2. [Algebraic Effects for Functional Programming](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/08/algeff-tr-2016-v2.pdf)
3. [Effekt: Extensible Algebraic Effects in Scala](https://files.b-studios.de/effekt.pdf)
4. [A Monadic Framework for Delimited Continuations](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.72.8645&rep=rep1&type=pdf)
5. [Implementing First-Class Polymorphic Delimited Continuations by a Type-Directed Selective CPS-Transform](https://infoscience.epfl.ch/record/149136/files/icfp113-rompf.pdf)
6. [Delimited continuations, macro expressiveness, and effect handlers](https://shonan.nii.ac.jp/archives/seminar/103/wp-content/uploads/sites/122/2016/09/effdel.pdf)
7. [scala-effekt](https://github.com/b-studios/scala-effekt)
8. [scala-continuations](https://github.com/scala/scala-continuations)

