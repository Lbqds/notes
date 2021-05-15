# recursion schemes

recursion schemes是对通用的递归模式的抽象，使得显式的递归变为隐式的递归，即我们不再需要定义递归的数据结构及递归的处理函数，递归部分完全由recursion schemes执行。patrickt在[这里](https://github.com/patrickt/recschemes/tree/master/org)对常见的recursion schemes及在haskell的开源实现做了详细的介绍，这篇文章的主要目的为:

* more examples: 对patrickt的介绍补充额外的例子，使得能够对recursion schemes有更直观的理解
* explains: 详细说明patrickt的介绍中存在的问题及需要注意的地方
* real-world examples: 介绍一些使用到recursion schemes的开源项目

本文所有的代码都可以在[这里](https://github.com/lbqds/recschemes)找到.

下面首先补充一些例子.

## paramorphism

相对于`cata`，`para`在fold的过程中除了记录当前fold的结果，还提供了当前的上下文:

```scala
final case class Fix[F[_]](unfix: F[Fix[F]])

type RAlgebra[F, A] = F[(Fix[F], A)] => A

def para[F[_]: Functor, A](ralgebra: RAlgebra[F, A])(fix: Fix[F]): A = {
  def f(fix: Fix[F]): (Fix[F], A) = (fix, para(ralgebra)(fix))
  ralgebra(fix.unfix.map(f))
}
```

patrickt给出了用`para`实现pretty print的例子，其中主要使用`para`来消除所有`id`函数的调用，Tim在[这里](https://www.youtube.com/watch?v=Zw9KeP3OzpU&ab_channel=LondonHaskell)提到了可以用`para`来实现sliding windows:

```scala
def slidingWindows[T](lst: List[T], size: Int): List[List[T]] = {
  val ralgebra: RAlgebra[ListF[T, *], List[List[T]]] = {
    case NilF => List.empty[List[T]]
    case ConsF(v, (curr, acc)) => 
      val lst = toList(curr)
      if (lst.size < size - 1) acc
      else (v +: lst.take(size - 1)) +: acc
  }
  para(ralgebra)(ListF(lst))
}
```

## apomorphism

`apo`与`para`相对立，相对于`ana`，`apo`在unfold的过程中可以提前决定是否结束unfold过程:

```scala
type RCoalgebra[F[_], A] = A => F[Either[Fix[F], A]]

def apo[F[_]: Functor, A](rcoalgebra: RCoalgebra[F, A])(a: A): Fix[F] =
  Fix(rcoalgebra(a).map {
    case Left(ctx) => ctx
    case Right(v) => apo(rcoalgebra)(v)
  })
```

同样是Tim在[这里](https://www.youtube.com/watch?v=Zw9KeP3OzpU&ab_channel=LondonHaskell)提到了通过`apo`来实现insert sort:

首先我们需要定义一些辅助数据结构:

```scala
sealed trait ListF[+E, +A]
final case object NilF extends ListF[Nothing, Nothing]
final case class ConsF(h: E, tail: A) extends ListF[E, A]

def nil[E, A]: ListF[E, A] = NilF

import cats.Functor

implicit def listFunctor[E]: Functor[ListF[E, *]] = new Functor[ListF[E, *]] {
  override def map[A, B](fa: ListF[E, A])(f: A => B): ListF[E, B] = fa match {
    case NilF => nil
    case ConsF(v, t) => ConsF(v, f(t))
  }
}
```

```scala
import scala.math.Ordering.Implicits._
def insertElement[T: Ordering]: Algebra[ListF[T, *], List[T]] = {
  val rcoalgebra: RCoalgebra[ListF[T, *], ListF[T, List[T]]] = {
    case NilF => nil
    case ConsF(v, t) if t.isEmpty => ConsF(v, Right(nil))
    case ConsF(v, l @ (h :: t)) if v <= h => ConsF(v, Left(ListF[T](l)))
    case ConsF(v, l @ (h :: t)) if v > h => ConsF(h, Right(ConsF(v, t)))
    case _ => ??? // never happen
  }
  (lst: ListF[T, List[T]]) => toList(apo(rcoalgebra)(lst))
}

def insertSort[T: Ordering](lst: List[T]): List[T] = cata(insertElement[T])(ListF[T](lst))
```

`insertSort`中，`cata`相当于foldRight，从`lst`的最右端开始依次执行`insertElement`，如果当前值比右边的小，则直接退出unfold；如果当前值比右边大，调换位置并继续执行unfold，直到插入到相对应的位置。

## hylomorphism

`hylo`是先unfold然后在fold的过程:

```scala
def hylo[F[_]: Functor, A, B](algebra: Algebra[F, B])(coalgebra: Coalgebra[F, A])(a: A): B
```

从类型不难看出，我们可以直接通过`cata`和`ana`来实现`hylo`:

```scala
def hylo[F[_]: Functor, A, B](algebra: Algebra[F, B])(coalgebra: Coalgebra[F, A])(a: A): B = cata(algebra)(ana(coalgebra)(a))
```

即我们先通过`ana`获得一个递归的数据结构，然后通过`cata`执行fold，这种实现方式并不高效，除了patrickt提到的O(n)和O(2n)的区别之外，在scala的中上述实现也*可能*会生成额外的临时数据，具体依赖实现，
比如在[droste](https://github.com/higherkindness/droste)的实现中`Fix[F]`会在编译时通过macro直接替换为`F`；而在[matryoshka](https://github.com/precog/matryoshka)中`Fix[F]`是`case class`。因此我们可以直接通过`algebra`和`coalgebra`来实现`hylo`:

```scala
def hylo[F[_]: Functor, A, B](algebra: Algebra[F, B])(coalgebra: Coalgebra[F, A])(a: A): B = 
  algebra(coalgebra(a).map(v => hylo(algebra)(coalgebra)(v)))
```

即我们先通过`coalgebra`展开一层运算，然后递归的调用`hylo`，最后通过`algebra`得到结果。

针对`hylo`，patrickt给出了逆波兰表达式的例子，其中有一点需要特别注意，即逆波兰表达式的计算为foldLeft，而cata的实现是foldRight，因此这里很巧妙的利用了higher carrier type `List[Int] => List[Int]`来执行计算过程。

matryoshka的例子中给出了用hylo来实现merge sort，merge sort的思想为将两个排序的列表合并成一个排序列表，我们可以将merge sort看作两个过程，第一个过程为将原始无序列表(假设类型为List[T])unfold为List[List[T]]，其中内层的每个列表只有一个元素，然后将这个列表的列表通过merge函数refold为一个顺序列表，整个过程代码如下:

我们首先定义一个非递归结构的Tree:

```scala
sealed trait TreeF[E, A]
final case class Leaf[E, A](v: E) extends TreeF[E, A]
final case class Branch[E, A](l: A, r: A) extends TreeF[E, A]
```

我们将无序列表unfold为`TreeF`，其中每个列表中的元素对应一个叶子节点，下面为`TreeF`来定义Functor:

```scala
import cats.Functor

implicit def treeFunctor[E]: Functor[TreeF[E, *]] = new Functor[TreeF[E, *]] {
  override def map[A, B](fa: TreeF[E,A])(f: A => B): TreeF[E,B] = fa match {
    case Leaf(v) => Leaf(v)
    case Branch(l, r) => Branch(f(l), f(r))
  }
}
```

有了上面基础的数据类型及functor之后，可以用其来实现merge sort:

```scala
import scala.math.Ordering.Implicits._
private def mergeLists[T: Ordering](l: List[T], r: List[T]): List[T] = (l, r) match {
  case (Nil, r) => r
  case (l, Nil) => l
  case (lh :: lt, rh :: rt) => if (lh < rh) lh +: mergeLists(lt, r) else rh +: mergeLists(l, rt)
}

import cats.data.NonEmptyList
def mergeSort[T: Ordering](lst: NonEmptyList[T]): List[T] = {
  val coalgebra: Coalgebra[TreeF[T, *], List[T]] = {
    case Nil => ??? // never happen
    case v :: Nil => TreeF.Leaf(v)
    case lst => 
      val (l, r) = lst.splitAt(lst.size / 2)
      TreeF.Branch(l, r)
  }
  val algebra: Algebra[TreeF[T, *], List[T]] = {
    case TreeF.Leaf(v) => List(v)
    case TreeF.Branch(l, r) => mergeLists(l, r)
  }
  hylo(algebra)(coalgebra)(lst.toList)
}
```

## histomorphism

相对于`cata`，`histo`在fold的执行过程中出了记录fold的结果，还保存了之前fold的历史记录:

```scala
// current accumulated value and history
final case class Attr[F[_], A](acc: A, his: F[Attr[F, A]])

type CVAlgebra[F[_], A] = F[Attr[F, A]] => A

def histo[F[_]: Functor, A](cvalgebra: CVAlgebra[F, A])(fix: Fix[F]): A = {
  val algebra: Algebra[F, Attr[F, A]] = (fattr: F[Attr[F, A]]) => Attr(cvalgebra(fattr), fattr)
  cata(algebra)(fix).acc
}
```

这里我们通过`cata`来实现`histo`，通过`algebra`我们可以很清楚的看到将整个fold的历史记录通过`Attr`来保存， `histo`在fold的过程中保存两个值，一个是当前fold的结果`acc`，另外一个是fold的历史记录`his`。 我们可以发现`Attr`其实就是Cofree Comonad，下面我们还会介绍如何通过`his`来去查找历史记录。

首先我们显举一个简单的例子，通过histo计算斐波那契数列(在这里我们省略了`Nat`及其functor的定义):

```scala
def fibByHisto(nat: Fix[Nat]): Int = {
  val cvalgebra: CVAlgebra[Nat, Int] = {
    case Zero => 0
    case Next(Attr(0, his)) => 1
    case Next(Attr(acc, his)) => his match {
      case Next(Attr(prev, _)) => acc + prev
      case Zero => ???   // never happen
    }
  }
  histo(cvalgebra)(nat)
}
```

我们可以看到，为了计算当前的值，我们通过查找历史记录，找到`fib(n - 1)`和`fib(n - 2)`，然后得到`fib(n)`。

patrickt给出了通过histo来实现`change making`的例子，很遗憾的是这个例子是**有问题的**，感兴趣的同学可以将patrickt的代码及递归版本的实现做一个对比。但通过histo来解决一些dynamic programming的思路是没有问题的，下面我们给出通过histo来实现unbounded knapsack问题的例子。

```scala
final case class Item(weight: Int, value: Double)

@scala.annotation.tailrec
private def lookup[A](cache: Attr[Nat, A], idx: Int): A = {
  if (idx == 0) cache.acc
  else cache.his match {
    case Next(attr) => lookup(attr, idx - 1)
    case _ => throw new IllegalAccessException(s"lookup index $idx large than cache size")
  }
}

def knapsackByHisto(capacity: Int, items: List[Item]): Double = {
  val cvalgebra: CVAlgebra[Nat, Double] = {
    case Zero => 0.0
    case nat @ Next(attr) =>
      val current = toInt(nat)
      val results = items.filter(_.weight <= current).map { item =>
        val remain = current - item.weight
        lookup(attr, current - 1 - remain) + item.value
      }
      if (results.isEmpty) 0.0 else results.max
  }
  histo(cvalgebra)(fromIntByAna(capacity))
}
```

这里我们通过`lookup`来查询历史记录，其中需要特别注意的是因为fold是从`Zero`开始的，因此lookup的索引为`current - 1 -remain`。

## dynamorphism

patrickt的介绍中没有提到`dyna`，`dyna`可以用来解决dynamic programming，[Histo- and Dynamorphisms Revisited](https://www.cs.ox.ac.uk/ralf.hinze/publications/WGP13.pdf)这篇论文重点介绍了`dyna`。这篇论文首先介绍如何通过`histomorphism`解决论文中列举的三个例子，然后说明了`histomorphism`的缺陷，即无法处理coinductive type，比如上述我们在用`histo`解决unbounded knapsack问题时，我们使用了`val current = toInt(nat)`来获取当前的值，而如果是coinductive type(通常是无限的)，我们没有办法通过其他的手段得到当前的值，因此`histo`在无法处理这种情况，但在论文中作者给出了另外一种通过`histo`不用获取当前的值来解决unbounded knapsack问题:

```scala
@scala.annotation.tailrec
private def lookup[A](cache: Attr[Nat, A], idx: Int): Option[A] = {
  if (idx == 0) Some(cache.acc)
  else cache.his match {
    case Next(attr) => lookup(attr, idx - 1)
    case _ => None
  }
}

def knapsackByHisto(capacity: Int, items: List[Item]): Double = {
  val cvalgebra: CVAlgebra[Nat, Double] = {
    case Zero => 0.0
    case Next(cache) => items.map(item =>
      lookup1(cache, item.weight - 1).map(_ + item.value).getOrElse(0.0)
    ).max
  }
  histo(cvalgebra)(fromIntByAna(capacity))
}
```

这里之所以能够正常运行是因为，在处理每一个`Item`时，我们需要的是`maximum(capacity - item.weight)`，而lookup的查找顺序是从当前位置开始查找，在缓存`Attr`中的index为`capacity - 1 - (capacity - item.weight)`，即`item.weight - 1`，当current大于`item.weight`时，我们就能够通过lookup找到对应的值，否则返回`None`，因此即便不知道当前值也可以正常运行。

但对于论文中的另外两个例子必须通过某种方式来获取到当前的值才可以正常工作，因此作者引出了`dyna`:

```scala
def dyna[F[_]: Functor, A, C](cvalgebra: CVAlgebra[F, A])(coalgebra: C => F[C])(c: C): A = {
  def helper: C => Attr[F, A] = (c: C) => {
    val x = coalgebra(c).map(v => helper(v))
    Attr(cvalgebra(x), x)
  }
  helper(c).acc
}
```

`dyna`可以理解为，首先通过`coalgebra`展开一层计算，然后递归的执行内层的fold。论文中作者描述了使用不同的思路通过`dyna`来解决另外的两个问题，感兴趣的同学可以看一看论文，上述的链接中也给出了相应的scala版本的实现。如果不好理解的话可以通过具体例子来去一步一步推导整个执行过程。

## matryoshka & droste

[matryoshka](https://github.com/precog/matryoshka)和[droste](https://github.com/higherkindness/droste)是两个开源的scala recursion schemes实现，两者都或多或少地借鉴了haskell [recursion schemes](https://github.com/recursion-schemes/recursion-schemes)的实现，特别是matryoshka，跟haskell版本的实现几乎是一样的，感兴趣的同学可以去看看具体的实现，在阅读源代码的过程中最好搭配一些实际的例子，这样能够帮助理解具体是怎么运行的。

## real-world examples

下面以[mu-scala](https://github.com/higherkindness/mu-scala)为例，介绍recursion schemes的具体应用，`mu-scala`是一个用于构建微服务的函数式库，它可以根据定义的protobuf文件生成相应的scala函数式代码，[这里](https://higherkindness.io/mu-scala/tutorials/service-definition/protobuf)有一个简单的示例，在使用过程中还涉及到下面两个组件:

* [skeuomorph](https://github.com/higherkindness/skeuomorph)

  所有的代码生成工作都在这个库中，`skeuomorph`中定义了三种schema，分别是`AvroF`、`ProtobufF`和`MuF`，前两个分别对应于avro和protobuf格式，并且可以通过recursion schemes直接转换为`MuF`，然后`skeuomorph`会使用recursion schemes对`MuF`进行优化，最后直接通过`scalameta`将`MuF`转换为scala代码。

* [sbt-mu-srcgen](https://github.com/higherkindness/sbt-mu-srcgen)是sbt插件，便于在命令行中生成代码，这个插件会直接调用`skeuomorph`.

对于每个grpc service定义，`skeuomorph`会生成一个带有`@service`标注的scala trait，而`@service`标注会在编译时通过scala macro生成一些辅助的代码便于开发使用，`@service`标注的实现在`mu-scala`中。

下面列出在`skeuomorph`中用到recursion schemes的地方(只列出了protobuf相关的)，感兴趣的同学可以看一下具体的实现:

* [pretty print](https://github.com/higherkindness/skeuomorph/blob/main/src/main/scala/higherkindness/skeuomorph/protobuf/print.scala): 通过`cata`实现schema的pretty print
* [parser](https://github.com/higherkindness/skeuomorph/blob/main/src/main/scala/higherkindness/skeuomorph/protobuf/ParseProto.scala): parser将protobuf源文件直接调用[protoc](https://github.com/os72/protoc-jar)的接口，然后利用`ana`将其转换为`ProtobufF`schema
* [ProtobufF to MuF](https://github.com/higherkindness/skeuomorph/blob/main/src/main/scala/higherkindness/skeuomorph/mu/Transform.scala): 利用transformer将`ProtobufF`转换为`MuF`
* [Optimize](): 同样利用transformer对`MuF`进行优化，即从`MuF`转换为`MuF`

除了`mu-scala`之外，[quasar](https://github.com/precog/quasar)中很多地方都用到了recursion schemes，后面有时间再补充。

## references

1. https://github.com/patrickt/recschemes/tree/master/org 
2. https://www.youtube.com/watch?v=Zw9KeP3OzpU&ab_channel=LondonHaskell
3. https://www.cs.ox.ac.uk/ralf.hinze/publications/WGP13.pdf
4. https://github.com/higherkindness/droste
5. https://github.com/precog/matryoshka
6. https://github.com/higherkindness/skeuomorph
7. https://github.com/precog/quasar
