package com.thoughtworks.deeplearning.etl

import java.io.File
import java.nio.{ByteBuffer, MappedByteBuffer}
import java.nio.channels.{Channels, FileChannel}
import java.nio.file._
import java.net.URL

import com.thoughtworks.future._
import com.thoughtworks.raii.asynchronous._
import org.rauschig.jarchivelib.{Archiver, ArchiverFactory}
import com.thoughtworks.each.Monadic._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.util.Random
import scalaz.syntax.all._

/**
  * @author 杨博 (Yang Bo)
  */
final case class Cifar100(trainBuffer: MappedByteBuffer, testBuffer: MappedByteBuffer) {
  import Cifar100._

  type CoarseClass = Int
  type GlobalFineClass = Int
  type LocalFineClass = Int

  private def numberOfTrainSamples = trainBuffer.capacity / NumberOfBytesPerSample

  private def numberOfTestSamples = testBuffer.capacity / NumberOfBytesPerSample

  private val globalFineClassesByLocalFineClass: Map[(CoarseClass, LocalFineClass), GlobalFineClass] = {
    val allClasses: Set[(CoarseClass, GlobalFineClass)] = {
      (0 until numberOfTestSamples).map { i =>
        val offset = i * NumberOfBytesPerSample
        val coarseClass = testBuffer.get(offset) & 0xFF
        val fineClass = testBuffer.get(offset + 1) & 0xFF
        (coarseClass, fineClass)
      }(collection.breakOut(Set.canBuildFrom))
    }
    val classMap: Map[CoarseClass, Set[(GlobalFineClass, LocalFineClass)]] = {
      allClasses
        .groupBy {
          case (coarseClass, globalFineClass) => coarseClass
        }
        .mapValues { subclasses =>
          subclasses.map {
            case (coarseClass, globalFineClass) => globalFineClass
          }.zipWithIndex
        }
    }
    for {
      (coarseClass, submap) <- classMap
      (globalFineClass, localFineClass) <- submap
    } yield (coarseClass, localFineClass) -> globalFineClass
  }

  private val localFineClassesByGlobalFineClass: Map[GlobalFineClass, (CoarseClass, LocalFineClass)] = {
    for ((path, globalFineClass) <- globalFineClassesByLocalFineClass) yield globalFineClass -> path
  }

  private def batchOneHotEncoding(label: Array[Int], numberOfClasses: Int): INDArray = {
    import org.nd4s.Implicits._
    val batchSize = label.length
    val encoded = Nd4j.zeros(batchSize, numberOfClasses)
    for (i <- label.indices) {
      encoded(i, label(i)) = 1
    }
    encoded
  }

  def epoch(batchSize: Int): Iterator[Batch] = {
    Random.shuffle[Int, IndexedSeq](0 until numberOfTestSamples).grouped(batchSize).map { batchIndices =>
      import org.nd4s.Implicits._
      val (coarseClasses, localFineClasses, pixels) = (for (trainImageIndex <- batchIndices) yield {
        val offset = trainImageIndex * NumberOfBytesPerSample
        val expectedCoarseClass = trainBuffer.get(offset) & 0xff
        val globalFineClass = trainBuffer.get(offset + 1) & 0xff
        val (coarseClass, localFineClass) = localFineClassesByGlobalFineClass(globalFineClass)
        if (coarseClass != expectedCoarseClass) {
          throw new IllegalStateException("Inconsistent coarse class detected in train data")
        }
        val imageArray = Array.ofDim[Byte](NumberOfPixelsPerSample)
        trainBuffer.position(offset + 2)
        trainBuffer.get(imageArray)
        val pixels = (for (pixel <- imageArray) yield {
          ((pixel & 0xff).toFloat + 0.5f) / 256.0f
        })(collection.breakOut(Array.canBuildFrom))
        (coarseClass, localFineClass, pixels)
      })(collection.breakOut(Array.canBuildFrom)).unzip3
      Batch(
        batchOneHotEncoding(coarseClasses, NumberOfCoarseClasses),
        batchOneHotEncoding(localFineClasses, NumberOfFineClassesPerCoarseClass),
        pixels.toNDArray.reshape(batchSize, NumberOfChannels, Width, Height)
      )
    }
  }
}

object Cifar100 {
  def main(args: Array[String]): Unit = {
    import com.thoughtworks.future._
    val cifar100 = load().blockingAwait
    val epoch = cifar100.epoch(50)
    epoch.next()
  }

  final case class Batch(coarseClasses: INDArray, localFineClasses: INDArray, pixels: INDArray)

  val Width = 32

  val Height = 32

  val NumberOfChannels = 3

  val NumberOfPixelsPerSample = Width * Height * NumberOfChannels

  val NumberOfLabelsPerSample = 2

  val NumberOfBytesPerSample = NumberOfLabelsPerSample + NumberOfPixelsPerSample

  val NumberOfCoarseClasses = 20

  val NumberOfFineClassesPerCoarseClass = 5

  val NumberOfFineClasses = NumberOfCoarseClasses * NumberOfFineClassesPerCoarseClass

  private val cacheDirectory = Paths.get(sys.props("user.home"), ".cifar100")

  private val extractedDataPath = cacheDirectory.resolve("cifar-100-binary")

  private def mapBuffer(path: Path) = {
    val channel = FileChannel.open(path, StandardOpenOption.READ)
    try {
      channel.map(FileChannel.MapMode.READ_ONLY, 0L, channel.size)
    } finally {
      channel.close()
    }
  }

  def load(url: URL = new URL("http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz")): Future[Cifar100] = {
    monadic[Future] {
      if (!Files.exists(extractedDataPath)) {
        val targzPath = Files.createTempFile("cifar-100-binary", ".tar.gz")
        def download =
          for {
            cifarHttpStream <- Do.autoCloseable(url.openStream())
            httpChannel <- Do.autoCloseable(Channels.newChannel(cifarHttpStream))
            fileChannel <- Do.autoCloseable(FileChannel.open(targzPath, StandardOpenOption.WRITE))
          } yield fileChannel.transferFrom(httpChannel, 0, Long.MaxValue)
        download.run.each
        val archiver: Archiver = ArchiverFactory.createArchiver("tar", "gz")
        archiver.extract(targzPath.toFile, cacheDirectory.toFile)
        extractedDataPath.ensuring(Files.exists(_))
      }

      val trainBuffer = mapBuffer(extractedDataPath.resolve("train.bin"))
      val testBuffer = mapBuffer(extractedDataPath.resolve("test.bin"))

      Cifar100(trainBuffer, testBuffer)
    }
  }
}
