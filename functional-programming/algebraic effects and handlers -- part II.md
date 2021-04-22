# Algebraic Effects and Handlers -- Part II

在第一部分中介绍了通过delimited continuation来实现algebraic effects，这部分主要介绍通过[freer monad](http://okmij.org/ftp/Haskell/extensible/more.pdf)来实现algebraic effects。

## Free Monad

在介绍freer monad之前先简单回顾一下free monad，free monad通过代码来表示为:

```scala
trait Free[F[_], A]
final case class Pure[F[_], A](x: A) extends Free[F, A]
final case class Impure[F[_], A](fa: F[Free[F, A]]) extends Free[F, A]
```

如果有`Functor[F]`，那么我们可以得到`Monad[Free[F, *]]`:

```scala
import cats.{Monad, Functor}

def monad[F: Functor]: Monad[Free[F, *]] = new Monad[Free[F, *]] {
  override def pure[A](x: A): Free[F,A] = Pure[F, A](x)
  override def flatMap[A, B](fa: Free[F,A])(f: A => Free[F,B]): Free[F,B] = fa match {
    case Pure(x) => f(x)
    case Impure(fa) => Impure[F, B](Functor[F].map(fa)(x => flatMap(x)(f)))
  }
  override def tailRecM[A, B](a: A)(f: A => Free[F,Either[A,B]]): Free[F,B] = ???
}
```

对于任意的functor，我们都可以通过free获得一个monad，在函数式编程中，free经常被用于构造eDSL。那么如果抛弃掉functor的限制，即对于任意的`F[A]`，我们是否都可以通过某种构造得到monad？答案就是通过freer monad。

**NOTE**: 由于[quadratic complexity](http://mandubian.com/2015/04/09/freer/), 上述free的实现并不实用，每次flatMap都会遍历整个结构，cats使用了额外的一个数据结构`FlatMapped`来[解决这个问题](https://typelevel.org/cats/datatypes/freemonad.html#for-the-very-curious-ones)。

## Freer Monad

freer monad实现如下所示:

```scala
trait Freer[F[_], A]
final case class Pure[F[_], A](x: A) extends Freer[F, A]
final case class Impure[F[_], A, B](fa: F[A], f: A => Freer[F, B]) extends Freer[F, B]

def functor[F[_]]: Functor[Freer[F, *]] = new Functor[Freer[F, *]] {
  override def map[A, B](fa: Freer[F,A])(f: A => B): Freer[F,B] = fa match {
    case Pure(x) => Pure[F, B](f(x))
    case Impure(fa1, f1) => Impure(fa1, (x: Any) => map(f1(x))(f))
  }
}

def monad[F[_]]: Monad[Freer[F, *]] = new Monad[Freer[F, *]] {
  override def pure[A](x: A): Freer[F,A] = Pure[F, A](x)
  override def flatMap[A, B](fa: Freer[F,A])(f: A => Freer[F,B]): Freer[F,B] = fa match {
    case Pure(x) => f(x)
    case Impure(fa1, f1) => Impure(fa1, (x: Any) => flatMap(f1(x))(f))
  }
  override def tailRecM[A, B](a: A)(f: A => Freer[F,Either[A,B]]): Freer[F,B] = ???
}
```

cats中对free monad的实现如下:

```scala
sealed abstract class Free[S[_], A] extends Product with Serializable with FreeFoldStep[S, A] {
  final def map[B](f: A => B): Free[S, B] =
    flatMap(a => Pure(f(a)))

  final def flatMap[B](f: A => Free[S, B]): Free[S, B] =
    FlatMapped(this, f)
}

final private[free] case class Pure[S[_], A](a: A) extends Free[S, A]
final private[free] case class Suspend[S[_], A](a: S[A]) extends Free[S, A]
final private[free] case class FlatMapped[S[_], B, C](c: Free[S, C], f: C => Free[S, B]) extends Free[S, B]

implicit def catsFreeMonadForFree[S[_]]: Monad[Free[S, *]] =
  new Monad[Free[S, *]] with StackSafeMonad[Free[S, *]] {
    def pure[A](a: A): Free[S, A] = Free.pure(a)
    override def map[A, B](fa: Free[S, A])(f: A => B): Free[S, B] = fa.map(f)
    def flatMap[A, B](a: Free[S, A])(f: A => Free[S, B]): Free[S, B] = a.flatMap(f)
  }
```

对比两个实现可以看到，cats中的free其实就是freer monad，即对于任意的`F[A]`，通过freer我们都可以得到monad。下面开始介绍freer monad在algebraic effects中的应用，主要内容来自于[Freer Monads, More Extensible Effects](http://okmij.org/ftp/Haskell/extensible/more.pdf)这篇论文。

extensible effects(或者algebraic effects)的一个主要的作用就是能够自由的组合不同的effect，而上述的freer monad仍然是针对某种单一的effect type，因此在这篇论文中作者引入了Open Union(`Union (r :: [* -> *]) a`)，其中`r`表示计算中所有可能出现的effect。

## extensible effects

由于这里使用的是scala语言，因此在实现上跟原论文有所不同，Open Union的实现如下所示:

```scala
trait Union {
  type Head[_]
  type Tail <: Union
}

trait :|:[F[_], U <: Union] extends Union {
  type Head[_] = F[_]
  type Tail = U
}

trait UNil extends Union {
  type Head[_] = Nothing
  type Tail = Nothing
}
```

除此之外，我们还需要额外的证明来表示某种effect type确实在Union中:

```scala
trait MemberIn[F[_], U <: Union] {
  val index: Int
}

object MemberIn {
  implicit def memberIn0[F[_], U <: Union]: MemberIn[F, F :|: U] = new MemberIn[F, F :|: U] {
    val index = 0
  }

  implicit def memberIn1[F[_], U <: Union](implicit m: MemberIn[F, U#Tail]): MemberIn[F, U] = new MemberIn[F, U] {
    val index = m.index + 1
  }

  implicit def memberNotIn0[F[_]]: MemberIn[F, UNil] = sys.error("effect type isn't a member of union")
  implicit def memberNotIn1[F[_]]: MemberIn[F, UNil] = sys.error("effect type isn't a member of union")
}
```

freer monad实现如下所示:

```scala
trait Freer[U <: Union, A] {
  def map[B](f: A => B): Freer[U, B] = Bind(this, (x: A) => Pure(f(x)))
  def flatMap[B](f: A => Freer[U, B]) = Bind(this, f)
  def >>=[B](f: A => Freer[U, B]) = flatMap(f)
  def >>[B](fb: Freer[U, B]): Freer[U, B] = Bind(this, (_: A) => fb)
}

final case class Pure[U <: Union, A](x: A) extends Freer[U, A]
private [freer] final case class Effect[U <: Union, A](inner: Any, index: Int) extends Freer[U, A]
final case class Bind[U <: Union, A, B](fa: Freer[U, A], f: A => Freer[U, B]) extends Freer[U, B]
```

在构造计算时，我们需要能够将某种effect注入进去:

```scala
def inject[F[_], U <: Union, A](fa: F[A])(implicit ev: MemberIn[F, U]): Effect[U, A] = Effect[U, A](fa, ev.index)
```

在解释执行时，我们需要按照在Union中的顺序依次执行interpreter，因此需要一个辅助方法从Union获取effectful computation:

```scala
def decompose[F[_], U <: Union, A](eff: Effect[U, A])(implicit ev: MemberIn[F, U]): Either[F[A], Effect[U#Tail, A]] = {
  if (ev.index == 0 && ev.index == eff.index) Left(eff.inner.asInstanceOf[F[A]])
  else Right(Effect[U#Tail, A](eff.inner, eff.index - 1))
}
```

`decompose`的返回值为`Either[F[A], Effect[U#Tail, A]]`，即或者是`F[A]`，或者是将`F`排除在外，因为我们是按照Union中的effect type的顺序来执行interpreter，因此每次都需要将index - 1。

我们可以将每种effect的解释执行抽象为interpreter:

```scala
trait Interpreter[F[_], U <: Union, A, B] {
  def onPure(x: A): Freer[U, B]
  def onEffect[T](eff: F[T], cont: T => Freer[U, B]): Freer[U, B]
}

def runInterpreter[U <: Union, F[_], A, B](
  program: Freer[U, A],
  interpreter: Interpreter[F, U#Tail, A, B]
)(
  implicit ev1: F[_] =:= U#Head[_],
  ev2: MemberIn[F, U]
): Freer[U#Tail, B] = program match {
  case Pure(x) => interpreter.onPure(x)
  case _: Effect[U, A] => throw new RuntimeException("internal error")
  case Bind(fa, f) => fa match {
    case Pure(x) => runInterpreter[U, F, A, B](f(x), interpreter)
    case eff @ Effect(inner, _) => decompose[F, U, Any](eff) match {
      case Left(e) => 
        interpreter.onEffect[Any](e, (x: Any) => runInterpreter[U, F, A, B](f(x), interpreter))
      case Right(r) =>
        r.flatMap((x: Any) => runInterpreter[U, F, A, B](f(x), interpreter))
    }
    case Bind(fa1, f1) => 
      val program1 = fa1.flatMap((x: Any) => f1(x).flatMap(f))
      runInterpreter[U, F, A, B](program1, interpreter)
  }
}
```

其中`runInterpreter`会使用`interpreter`来执行所有`program`中的effect `F`，返回的计算中将不会再包含`F`。

当执行完所有的effect时，最终的计算为`Freer[UNil, A]`:

```scala
implicit class RunOps[A](p: Freer[UNil, A]) {
  def run: A = p match {
    case Pure(x) => x
    case Bind(fa, f) => new RunOps(f(new RunOps(fa).run)).run
    case _ => throw new RuntimeException("internal error")
  }
}
```

在原论文中，作者还做了一些优化，比如上述的实现中跟前面Free Monad的实现有同样的性能问题，因此论文中通过队列来保存所有的continuation。

下面来看一些示例。

## examples

在[Algebraic Effects for Functional Programming](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/08/algeff-tr-2016-v2.pdf)的2.4节的例子中提到了不同的运行顺序会得到不同的结果，下面我们通过上述的实现来重现这个例子:

`State`的实现如下所示:

```scala
trait State[+S]
final case class Set[S](value: S) extends State[Unit]
final case class Get[S]() extends State[S]

object State {
  def set[U <: Union, S](init: S)(implicit ev: MemberIn[State, U]): Freer[U, Unit] = 
    Freer.send[State, U, Unit](Set(init))

  def get[U <: Union, S](implicit ev: MemberIn[State, U]): Freer[U, S] = 
    Freer.send[State, U, S](Get[S])

  implicit class StateOps[U <: Union, A](fa: Freer[U, A]) {
    def runState[S](init: S)(implicit ev1: State[_] =:= U#Head[_], ev2: MemberIn[State, U]): Freer[U#Tail, (A, S)] = fa match {
      case Pure(x) => Freer.pure[U#Tail, (A, S)]((x, init))
      case Effect(_, _) => throw new RuntimeException("internal error")
      case Bind(fa, f) => fa match {
        case Pure(x) => new StateOps(f(x)).runState(init)
        case eff @ Effect(_, _) => Freer.decompose[State, U, Any](eff) match {
          case Left(e) => e match {
            case Get() => new StateOps(f(init)).runState(init)
            case Set(value) => new StateOps(f()).runState(value.asInstanceOf[S])
          }
          case Right(r) => r.flatMap((x: Any) => new StateOps(f(x)).runState(init))
        }
        case Bind(fa1, f1) => 
          val program1 = fa1.flatMap((x: Any) => f1(x).flatMap(f))
          new StateOps(program1).runState(init)
      }
    }
  }
}
```

`Amb`的实现如下所示:

```scala
trait Amb[T]
final case object Flip extends Amb[Boolean]

object Amb {
  def flip[U <: Union](implicit ev: MemberIn[Amb, U]): Freer[U, Boolean] = 
    Freer.send[Amb, U, Boolean](Flip)

  implicit class AmbOps[U <: Union, A](fa: Freer[U, A]) {
    def runAmb(implicit ev1: Amb[_] =:= U#Head[_], ev2: MemberIn[Amb, U]): Freer[U#Tail, List[A]] = {
      Freer.runInterpreter[U, Amb, A, List[A]](fa, new Interpreter[Amb, U#Tail, A, List[A]] {
        override def onPure(x: A): Freer[U#Tail,List[A]] = Freer.pure[U#Tail, List[A]](List(x))

        override def onEffect[T](eff: Amb[T], cont: T => Freer[U#Tail,List[A]]): Freer[U#Tail,List[A]] = {
          val f = cont.asInstanceOf[Boolean => Freer[U#Tail, List[A]]]
          for {
            left  <- f(false)
            right <- f(true)
          } yield left ++ right
        }
      })
    }
  }
}
```

```scala
// Algebraic Effects for Functional Programming(2.4)
type ET = Amb :|: State :|: UNil
type _amb[U <: Union] = MemberIn[Amb, U]
def xor[U <: Union : _amb]: Freer[U, Boolean] = for {
  p <- Amb.flip[U]
  q <- Amb.flip[U]
} yield (p || q) && !(p && q)

type _state[U <: Union] = MemberIn[State, U]
def suprising[U <: Union : _amb : _state]: Freer[U, Boolean] = for {
  p <- Amb.flip[U]
  s <- State.get[U, Int]
  _ <- State.set[U, Int](s + 1)
  b <- if (s >= 1 && p) xor[U] else Freer.pure[U, Boolean](false)
} yield b

println(suprising[ET].runAmb.runState[Int](0).run)   // result: (List(false, false, true, true, false), 2)

type TE = State :|: Amb :|: UNil
println(suprising[TE].runState[Int](0).runAmb.run)   // result: List((false, 1), (false, 1))
```

我们可以看到结果与论文中的结果相同。下面我们简单介绍一些基于`Freer Monads, More Extensible Effects`这篇论文的开源实现。

* eff

[eff](https://github.com/atnos-org/eff)是scala版本的实现，不同于原论文及上述的实现，effect的运行顺序不需要跟Union中的声明顺序保持一致，`eff`在运行时会对continuation的参数重排序。在`eff`的实现中，[Continuation](https://github.com/atnos-org/eff/blob/master/shared/src/main/scala/org/atnos/eff/Continuation.scala#L19)类似于原论文中的`FTCQueue`。`eff`也实现了Applicative用于并行操作，但在使用时需要注意`eff`不满足Monad-Applicative law，在haskell中通常用`<*> = ap`来表示。

* freer-simple

[freer-simple](https://github.com/lexi-lambda/freer-simple)是haskell版本的实现，基本与论文中的一致。

## 不足之处

上述通过freer monad来实现algebraic effects仍然存在不足之处，下面看一个例子:

```scala
trait Exception[E]
final case class Throw[E <: Throwable](e: E) extends Exception[E]

object Exception {
  def `throw`[U <: Union, E <: Throwable](e: E)(implicit ev: MemberIn[Exception, U]): Freer[U, E] = 
    Freer.send[Exception, U, E](Throw(e))

  def `catch`[U <: Union, E <: Throwable, A](
    fa: Freer[U, A], h: E => Freer[U, A]
  )(implicit ev: MemberIn[Exception, U]): Freer[U, A] = fa match {
    case Effect(_, _) => throw new RuntimeException("internal error")
    case eff @ Pure(_) => eff
    case Bind(fa, f) => fa match {
      case Pure(x) => `catch`[U, E, A](f(x), h)
      case Effect(inner, _) => 
        if (inner.isInstanceOf[Exception[_]]) h(inner.asInstanceOf[Throw[E]].e)
        else fa.flatMap((x: Any) => `catch`[U, E, A](f(x), h))
      case Bind(fa1, f1) => 
        val program1 = fa1.flatMap((x: Any) => f1(x).flatMap(f))
        `catch`[U, E, A](program1, h)
    }
  }

  implicit class ExceptionOps[U <: Union, A](fa: Freer[U, A]) {
    def runExc[E <: Throwable](
      implicit ev1: Exception[_] =:= U#Head[_], 
      ev2: MemberIn[Exception, U]
    ): Freer[U#Tail, Either[E, A]] = {
      Freer.runInterpreter[U, Exception, A, Either[E, A]](
        fa, 
        new Interpreter[Exception, U#Tail, A, Either[E, A]] {
          override def onPure(x: A): Freer[U#Tail,Either[E,A]] = Freer.pure(Right(x))

          override def onEffect[T](eff: Exception[T], cont: T => Freer[U#Tail,Either[E,A]]): Freer[U#Tail,Either[E,A]] =
            Freer.pure(Left(eff.asInstanceOf[Throw[E]].e))
        }
      )
    }
  }
}

// Effect Handlers in Scope(8)
type _exc[U <: Union] = MemberIn[Exception, U]
def decr[U <: Union : _exc : _state]: Freer[U, Unit] = for {
  x <- State.get[U, Int]
  _ <- if (x > 0) State.set[U, Int](x - 1) else Exception.`throw`[U, Throwable](new Throwable("less then 0"))
} yield ()

def tripleDecr[U <: Union : _exc : _state]: Freer[U, Unit] = 
  decr[U] >> Exception.`catch`[U, Throwable, Unit](decr[U] >> decr[U], _ => Freer.pure[U, Unit](()))
```

对于`tripleDecr`，在执行时对于不同的执行顺序应该能够得到不同的结果，一种是state的结果是0(global interpretation，前两个decr执行)，另一种是state的结果是1(local interpretation，只有第一个decr执行)，现在来看看上述实现的结果:

```scala
type ET = State :|: Exception :|: UNil
println(tripleDecr[ET].runState[Int](2).runExc.run)

type TE = Exception :|: State :|: UNil
println(tripleDecr[TE].runExc.runState[Int](2).run)
```

输出结果为:

```scala
Right(((),0))
(Right(()),0)
```

得到的结果跟我们的期望不符，下面我们通过monad transformer来实现上述的示例:

```scala
import cats.mtl._
import cats.mtl.implicits._
import cats._
import cats.data._
import cats.implicits._

def decr[F[_]: Monad : Stateful[*[_], Int] : Handle[*[_], Throwable]]: F[Unit] = for {
  x <- Stateful[F, Int].get
  _ <- if (x > 0) Stateful[F, Int].set(x - 1) else Handle[F, Throwable].raise(new Throwable("less then 0"))
} yield ()

def tripleDecr[F[_]: Monad : Stateful[*[_], Int] : Handle[*[_], Throwable]]: F[Unit] =
  decr[F] >> Handle[F, Throwable].handleWith(decr[F] >> decr[F])(_ => Monad[F].unit)

println(tripleDecr[StateT[EitherT[Id, Throwable, *], Int, *]].run(2).value)
println(tripleDecr[EitherT[StateT[Id, Int, *], Throwable, *]].value.run(2))
```

上述输出结果为:

```scala
Right((1,()))
(0,Right(()))
```

我们发现，通过monad transformer可以得到我们期望的结果。上述示例来源于[Effect Handlers in Scope](https://www.cs.ox.ac.uk/people/nicolas.wu/papers/Scope.pdf)这篇论文，主要原因在于上述algebraic effects的实现中，handler不仅提供了语义，同时还限定了effect作用的范围，在Effect Handlers in Scope中，作者通过引入higher-order effects来解决这个问题，关于higher-order effects下一部分再详细介绍。

上述所有的代码在[这里](https://github.com/lbqds/freer)。

## 其他

对比haskell的实现，我们可以发现通常在scala的代码中需要很多额外的类型声明，主要是因为haskell跟scala所使用的类型推导算法不同，haskell中使用的类型推导算法为global type inference，而scala由于支持子类型多态的缘故采用的是local type inference算法。感兴趣的可以参考[这里](https://madusudanan.com/blog/scala-tutorials-part-2-type-inference-in-scala/)。

## reference

* [Freer Monads, More Extensible Effects](http://okmij.org/ftp/Haskell/extensible/more.pdf)
* [Algebraic Effects for Functional Programming](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/08/algeff-tr-2016-v2.pdf)
* [Effect Handlers in Scope](https://www.cs.ox.ac.uk/people/nicolas.wu/papers/Scope.pdf)
* https://atnos-org.github.io/eff/index.html
* https://github.com/atnos-org/eff
* https://github.com/lexi-lambda/freer-simple
