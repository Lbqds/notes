## metascala

前段实现看了lihaoyi写的[metascala](https://github.com/lihaoyi/metascala), 这里简单做个总结。下面我主要通过对象内存布局、内存分配、
gc、VM、代码执行几个部分具体介绍metascala的实现。

### Type

metascala中将类型分为`Prim`和`Ref`两种类型(Type.scala):

```scala
/**
 * Represents all variable types within the Metascala VM
 */
trait Type{
  /**
   * The JVMs internal representation
   * - V Z B C S I F J D
   * - Ljava/lang/Object; [Ljava/lang/String;
   */
  def internalName: String

  /**
   * Nice name to use for most things
   * - V Z B C S I F J D
   * - java/lang/Object [java/lang/String
   */
  def name: String

  override def toString = name

  /**
   * A human-friendly name for this type that can be used everywhere without
   * cluttering the screen.
   * - V Z B C S I F J D
   * - j/l/Object [j/l/String
   */
  def shortName = shorten(name)

  /**
   * The thing that's returned by Java's getName method
   * - void boolean byte char short int float long double
   * - java.lang.Object [java.lang.String;
   */
  def javaName: String
  /**
   * Retrieves the Class object in the host JVM which represents the
   * given Type inside the Metascala VM
   */
  def realCls: Class[_]

  /**
   * Reads an object of this type from the given input stream into a readable
   * representation
   */
  def prettyRead(x: () => Val): String

  /**
   * 0, 1 or 2 for void, most things and double/long
   */
  def size: Int

  def isRef: Boolean = this.isInstanceOf[imm.Type.Ref]
}
```

注意`size`表示大小，由于metascala中内存使用Int(下面会提到)表示，所以对于Ref类型和除Double/Long之外的Prim类型，大小都为4个字节。

### 内存分配

在看内存分配之前需要了解metascala中Object和Array的布局，对于Object:

size(Object) = sum(header size, fields size)

其中，header size为8个字节，前4个字节存储class的索引负数，即class在`ClsTable.clsIndex`(下面会讲到ClsTable)的索引的相反数，负数是为了区分引用是Object还是Array。
后4个字节存储fields length，分配Object的代码如下(Obj.scala):

```scala
val headerSize = 2
def allocate(cls: rt.Cls, initMembers: (String, Ref)*)(implicit registrar: Registrar): Obj = {
  println(s"alloc ${cls.name}")
  implicit val vm = registrar.vm
  val address = vm.heap.allocate(headerSize + cls.fieldList.length)
  vm.heap(address()) = -cls.index
  vm.heap(address() + 1) = cls.fieldList.length
  val obj = new Obj(address)
  for ((s, v) <- initMembers){
    obj(s) = v()
  }
  obj
}
```

NOTE: 这里`headerSize`定义为2，但在Heap中实际内存的存储单位为Int，因此`headerSize`为8个字节。即实际的内存为`(headerSize + cls.fieldList.length) * 4`字节

对于Array:

size(array) = sum(header size, elements size)

其中header size为8个字节，前4个字节存储数组类型索引，即type在`ClsTable.arrayTypeCache`的索引，后4个字节存储数组的长度(Obj.scala):

```scala
val headerSize = 2
/**
 * Allocates and returns an array of the specified type, with `n` elements.
 * This is multiplied with the size of the type being allocated when
 * calculating the total amount of memory benig allocated
 */
def allocate(t: imm.Type, n: scala.Int)(implicit registrar: Registrar): Arr = {
  rt.Arr.allocate(t, Array.fill[Int](n * t.size)(0).map(x => new ManualRef(x): Ref))
}
def allocate(innerType: imm.Type, backing: Array[Ref])(implicit registrar: Registrar): Arr = {
  implicit val vm = registrar.vm
  val address = vm.heap.allocate(Arr.headerSize + backing.length)
  vm.heap(address()) = vm.arrayTypeCache.length
  vm.arrayTypeCache.append(innerType)
  vm.heap(address() + 1) = backing.length / innerType.size
  // backing init with pos 0, when call Arr[i] = xx; backing would be updated
  backing.map(_()).copyToArray(vm.heap.memory, address() + Arr.headerSize)
  new Arr(address)
}
```

NOTE: 这里`headerSize`定义为2，但在Heap中实际内存的存储单位为Int，因此`headerSize`为8个字节。即实际的内存为`(headerSize + backing.length) * 4`字节

内存分配采用sequential allocation(Heap.scala):

```scala
val memory = new Array[Int](memorySize * 2)
var start = 0
var freePointer = 1
def allocate(n: Int)(implicit registrar: Registrar) = {
  if (freePointer + n > memorySize + start) {
    println("COLLECT LOL")
    collect(start)
    if (freePointer + n > memorySize + start) {
      throw new Exception("Out of Memory!")
    }
  }
  val newFree = freePointer
  freePointer += n
  val ref = new ManualRef(newFree)
  registrar(ref)
  ref
}
```

`Heap`维护一个连续的内存块`memory`，注意这里为`Array[Int]`类型，即`memory`大小为(memorySize * 2) * 4字节。

这里的`memorySize * 2`是因为metascala采用semispace copying gc算法，后面会详细描述。

`Heap`维护一个`freePointer`，每次分配时判断剩余空间是否满足需求，如果不满足则调用gc(`collect`)。这里的implicit参数registrar用于保存已分配
的但还无法通过stack roots索引到的对象。

`allocate`分配的结果为`ManualRef`类型(`metascala/package.scala`):

```scala
implicit class ManualRef(var x: Int) extends Ref{
  def apply() = x
  def update(i: Int) = x = i
}
```

其中x为对象在堆中的地址，update用于gc时修改Ref的地址。

### GC

在分配内存时，如果剩余内存不足就会触发gc，metascala采用semispace copying gc算法，将内存分为大小相等的两部分，其中一部分时空闲的，
当gc触发时，从stack root索引所有对象，并将所有对象移动到另外一半，同时修改ref地址，然后更新freePointer。代码如下(Heap.scala):

```scala
def blit(freePointer: Int, src: Int) = {
  if (src == 0 || memory(src) == 0) {
    (0, freePointer)
  } else if (memory(src) == 0){
    throw new Exception("Can't point to nothing! " + src + " -> " + memory(src))
  } else if (memory(src + 1) >= 0){                                                           
    val headerSize = if (isObj(memory(src))) rt.Obj.headerSize else rt.Arr.headerSize
    val length =  memory(src+1) + headerSize                       // 计算对象所占内存大小，并将整个对象copy到freePointer
    System.arraycopy(memory, src, memory, freePointer, length)
    memory(src + 1) = -freePointer                                 // 将原地址起始4字节处标记为-freePointer，表示当前对象已经被copy
    (freePointer, freePointer + length)                            // 返回新的ref地址和更新后的freePointer
  } else {
    println(s"non-blitting $src -> ${-memory(src + 1)}")           
    (-memory(src + 1), freePointer)                                // memory(src + 1) < 0表示当前对象已经被copy，直接返回新的地址，并且freePointer不变
  }
}

def collect(from: Int){
  val to = (from + memorySize) % (2 * memorySize)                       // 计算copy的目标起始位置，即另一半空闲空间的起始位置
  for(i <- to until to+memorySize){
    memory(i) = 0                                                       // memset zero
  }
  println("===============Collecting==================")
  println("starting " + (freePointer - from))
  val roots = getRoots()                                                // stack roots，refer to VM.getRoots()
  var scanPointer = 0
  if (from == 0){                                                       // 初始化freePointer和scanPointer为空闲部分的起始位置
    freePointer = memorySize + 1
    scanPointer = memorySize + 1
  } else{ // to == memorySize
    start = 0
    freePointer = 1
    scanPointer = 1
  }

  for (root <- roots) {                                                // 遍历root sets，
    val oldRoot = root()
    val (newRoot, nfp) = blit(freePointer, oldRoot)                    // 调用blit获取新的ref的地址和新的freePointer 
    freePointer = nfp                                                  // 更新freePointer
    root() = newRoot                                                   // 更新root ref地址
  }

  // 因为上面的for循环只处理root sets，所以需要从scanPointer开始遍历所有存活的对象，然后将其copy到新的freePointer，
  // 当scanPointer与freePointer相等时说明所有的对象都已经被copy到新的地址
  while(scanPointer != freePointer) {
    assert(scanPointer <= freePointer, s"scanPointer $scanPointer > freePointer $freePointer")

    val links = getLinks(memory(scanPointer), memory(scanPointer+1))
    val length = memory(scanPointer + 1) + rt.Obj.headerSize

    for(i <- links){
      val (newRoot, nfp) = blit(freePointer, memory(scanPointer + i))
      memory(scanPointer + i) = newRoot
      freePointer = nfp
    }

    scanPointer += length
  }

  if (from == 0) start = memorySize                          // 最后更新start
  else start = 0

  println("ending " + (freePointer - start))

  println("==================Collectiong Compelete====================")
}
```

由于需要copy到新的地方，因此对象引用的地址就会改变，需要将对象的引用更新到指向新的地址。即`root() = newRoot`和`memory(scanPointer + i) = newRoot`这两行代码。

### VM

VM用于存储所有运行时需要的信息，包括Heap，type cache，class table等，其中`ClsTable`(VM.scala)用于存储所有已加载的class:

```scala
implicit object ClsTable extends Cache[imm.Type.Cls, rt.Cls]{
  val clsIndex = mutable.ArrayBuffer[rt.Cls](null)

  def calc(t: imm.Type.Cls): rt.Cls = {
    val input = natives.fileLoader(
      t.name + ".class"
    ).getOrElse(
      throw new Exception("Can't find " + t)
    )
    val cr = new ClassReader(input)
    val classNode = new ClassNode()
    cr.accept(classNode, ClassReader.EXPAND_FRAMES)

    Option(classNode.superName).map(Type.Cls.read).map(vm.ClsTable)
    rt.Cls(classNode, clsIndex.length)
  }

  var startTime = System.currentTimeMillis()

  override def post(cls: rt.Cls) = {
    clsIndex.append(cls)
  }
}
```

class通过asm加载，并保存在`clsIndex`中，在通过asm加载之后会初始化`Cls`(Cls.scala):

```scala
def apply(cn: ClassNode, index: Int)(implicit vm: VM) = {
  import imm.NullSafe._
  val fields = cn.fields.safeSeq.map(imm.Field.read)
  val superType = cn.superName.safeOpt.map(Type.Cls.read)
  new Cls(
    tpe = imm.Type.Cls.read(cn.name),
    superType = superType,
    sourceFile = cn.sourceFile.safeOpt,
    interfaces = cn.interfaces.safeSeq.map(Type.Cls.read),
    accessFlags = cn.access,
    methods =
      cn.methods
        .safeSeq
        .zipWithIndex
        .map{case (mn, i) =>
        new rt.Method.Cls(
          index,
          i,
          Sig(mn.name, imm.Desc.read(mn.desc)),
          mn.access,
          () => Conversion.ssa(cn.name, mn)
        )
      },
    fieldList =
      superType.toSeq.flatMap(_.cls.fieldList) ++
        fields.filter(!_.static).flatMap{x =>
          Seq.fill(x.desc.size)(x)
        },
    staticList =
      fields.filter(_.static).flatMap{x =>
        Seq.fill(x.desc.size)(x)
      },
    outerCls = cn.outerClass.safeOpt.map(Type.Cls.read),
    index
  )
}
```

其中会将所有method的代码转换成SSA的形式，代码位于(Conversion.scala)，代码太长这里就不贴了，我大致描述一下转换的流程。

`Box`(Conversion.scala)对应一个single assignment variable，在`AbstractFunnyInterpreter`中，也可以看到除了copy之外，所有其他的操作都会重新创建一个Box。

首先将method内的代码根据return/lookup switch/table switch/jump指令划分为多个code block，然后使用asm模拟执行计算local size，并将所有指令转换为`Insn`(Insn.scala)，
最后根据code block之间的跳转关系构造phi node。

### execute

执行方法调用时，对于InvokeVirtual，直接从`vTable`根据index获取方法，对于InvokeInterface，需要去`vTableMap`中查询方法:

```scala
/**
 * The virtual function dispatch table
 */
lazy val vTable: Seq[rt.Method] = {
  val oldMethods =
    mutable.ArrayBuffer(
      superType
             .toArray
             .flatMap(_.vTable): _*
    )
  methods.filter(!_.static)
         .foreach{ m =>
    val index = oldMethods.indexWhere{ mRef => mRef.sig == m.sig }
    val native = vm.natives.trapped.find{case rt.Method.Native(clsName, sig, func) =>
      (name == clsName) && sig == m.sig
    }
    val update: Method => Unit =
      if (index == -1) oldMethods.append(_: Method)
      else oldMethods.update(index, _: Method)
    native match {
      case None => update(m)
      case Some(native) => update(native)
    }
  }
  oldMethods
}
/**
 * A hash map of the virtual function table, used for quick lookup
 * by method signature
 */
lazy val vTableMap = vTable.map(m => m.sig -> m).toMap
```

对于InvokeStatic和InvokeSpecial，直接从`methods`中根据signature获取方法，metascala未实现InvokeDymaic指令。

每个方法调用会生成一个`Frame`:

```scala
/**
 * The stack frame created by every method call
 */
class Frame(var pc: (Int, Int) = (0, 0),
            val runningClass: rt.Cls,
            val method: rt.Method.Cls,
            var lineNum: Int = 0,
            val returnTo: Int => Unit,
            val locals: Array[Val])
```

其中`throwExWithTrace`在抛出异常时也会将调用栈(`threadStack`)打印出来:

```scala
final def throwExWithTrace(clsName: String, detailMessage: String) = {
  throwException(
    vm.alloc( implicit r =>
      rt.Obj.allocate(clsName,
        "stackTrace" -> trace.toVirtObj,
        "detailMessage" -> detailMessage.toVirtObj
      )
    )
  )
}
```

其中`trace`定义如下:

```scala
def trace = {
  threadStack.map( f =>
    new StackTraceElement(
      f.runningClass.name.toDot,
      f.method.sig.name,
      f.runningClass.sourceFile.getOrElse("<unknown file>"),
      try f.method.code.blocks(f.pc._1).lines(f.pc._2) catch{case _ => 0 }
    )
  ).toArray
}
```

其他指令执行相关的代码都在Thread.scala，这部分代码比较通俗易懂，这里就不再解释了。

### 总结

metascala作为一个极简的jvm实现，虽然只支持单线程，并且内存分配和gc算法都非常简单，但通过学习metascala的实现仍然可以很好的了解jvm的大致工作原理。
