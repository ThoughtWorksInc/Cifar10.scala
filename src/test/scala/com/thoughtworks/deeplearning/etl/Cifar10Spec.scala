package com.thoughtworks.deeplearning.etl

import org.scalatest.{FreeSpec, Matchers}
import com.thoughtworks.future._

class Cifar10Spec extends FreeSpec with Matchers {
  "Given some Cifar10 data" - {
    val cifar10: Cifar10 = Cifar10.load().blockingAwait

    "When load a batch that contains less number of indices than batch size" - {

      val indices = IndexedSeq(4, 100, 30)
      val batch = cifar10.loadBatch(5,indices )

      "Then the batch should contains the same number of samples with number of indices" in {
        batch.labels.shape().head should be(indices.length)
        batch.pixels.shape().head should be(indices.length)
      }

    }

  }
}
