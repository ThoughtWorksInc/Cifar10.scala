name := "cifar100"

organization := "com.thoughtworks.deeplearning.etl"

libraryDependencies += "org.rauschig" % "jarchivelib" % "0.5.0"

libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "0.8.0" % Runtime

libraryDependencies += "org.nd4j" % "nd4j-api" % "0.8.0"

libraryDependencies += "org.nd4j" %% "nd4s" % "0.8.0"

libraryDependencies += "com.thoughtworks.each" %% "each" % "3.3.1"

libraryDependencies += "com.thoughtworks.raii" %% "asynchronous" % "3.0.0-M6"

scalaVersion := "2.11.11"