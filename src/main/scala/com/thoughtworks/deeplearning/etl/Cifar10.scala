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

import scala.collection.{immutable, mutable}
import scala.util.Random
import scalaz.syntax.all._

/**
  * @author Rabenda
  */
final case class Cifar10(trainBuffers: Seq[MappedByteBuffer], testBuffer: MappedByteBuffer) {

  import Cifar10._


  /** file 里有多少个图像 */
  private def numberOfTrainSamplesPerFile = trainBuffers.head.capacity / NumberOfBytesPerSample

  if (trainBuffers.map(_.capacity).toSet.size != 1) {
    throw new IllegalArgumentException("Train files should not have different sizes.")
  }

  if (trainBuffers.head.capacity % NumberOfBytesPerSample != 0) {
    throw new IllegalArgumentException(s"Train files' size must be $NumberOfBytesPerSample multiple.")
  }

  /** total 图像数 */
  private def numberOfTrainSamples = numberOfTrainSamplesPerFile * trainBuffers.length


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
    Random.shuffle[Int, IndexedSeq](0 until numberOfTrainSamples).grouped(batchSize).map {
      batchIndices => loadBatch(batchSize, batchIndices)
    }
  }

  private def loadBatch(batchSize: Int, batchIndices: IndexedSeq[Int]): Batch = {

    import org.nd4s.Implicits._
    val (labels, pixels) = (for (trainImageIndex <- batchIndices) yield {
      val offset = trainImageIndex % numberOfTrainSamplesPerFile * NumberOfBytesPerSample
      val fileIndex = trainImageIndex / numberOfTrainSamplesPerFile

      val label = trainBuffers(fileIndex).get(offset) & 0xff
      val imageArray = Array.ofDim[Byte](NumberOfPixelsPerSample)


      trainBuffers(fileIndex).position(offset + 1)
      trainBuffers(fileIndex).get(imageArray)
      val pixels: Array[Float] = for (pixel <- imageArray) yield {
        ((pixel & 0xff).toFloat + 0.5f) / 256.0f
      }

      (label, pixels)
    }) (collection.breakOut(Array.canBuildFrom)).unzip

    Batch(
      batchOneHotEncoding(labels, NumberOfClasses),
      pixels.toNDArray.reshape(batchSize, NumberOfChannels, Width, Height)
    )
  }
}

object Cifar10 {

  final case class Batch(labels: INDArray, pixels: INDArray)

  val Width = 32

  val Height = 32

  val NumberOfChannels = 3

  val NumberOfPixelsPerSample = Width * Height * NumberOfChannels

  val NumberOfLabelsPerSample = 1

  val NumberOfBytesPerSample = NumberOfLabelsPerSample + NumberOfPixelsPerSample

  val NumberOfClasses = 10

  private val cacheDirectory = Paths.get(sys.props("user.home"), ".cifar10")

  private val extractedDataPath = cacheDirectory.resolve("cifar-10-batches-bin")

  private def mapBuffer(path: Path) = {
    val channel = FileChannel.open(path, StandardOpenOption.READ)
    try {
      channel.map(FileChannel.MapMode.READ_ONLY, 0L, channel.size)
    } finally {
      channel.close()
    }
  }

  def load(url: URL = new URL("http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz")): Future[Cifar10] = {
    monadic[Future] {
      if (!Files.exists(extractedDataPath)) {
        val targzPath = Files.createTempFile("cifar-10-binary", ".tar.gz")

        def download =
          for {
            cifarHttpStream <- Do.scoped(url.openStream())
            httpChannel <- Do.scoped(Channels.newChannel(cifarHttpStream))
            fileChannel <- Do.scoped(FileChannel.open(targzPath, StandardOpenOption.WRITE))
          } yield fileChannel.transferFrom(httpChannel, 0, Long.MaxValue)

        download.run.each
        val archiver: Archiver = ArchiverFactory.createArchiver("tar", "gz")
        archiver.extract(targzPath.toFile, cacheDirectory.toFile)
        extractedDataPath.ensuring(Files.exists(_))
      }


      val trainBuffers = for {
        i <- 1 to 5
      } yield {
        mapBuffer(extractedDataPath.resolve(s"data_batch_$i.bin"))
      }
      val testBuffer = mapBuffer(extractedDataPath.resolve("test_batch.bin"))

      Cifar10(trainBuffers, testBuffer)
    }
  }
}
