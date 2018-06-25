jarName in assembly := "stock_deep_learning_uber.jar"
mainClass in assembly := Some("com.frank.AnnApp")
version := "1.0"
scalaVersion := "2.11.11"

mergeStrategy in assembly ~= { (old) =>
  {
    case PathList("META-INF", "MANIFEST.MF") => MergeStrategy.discard
    case PathList("META-INF", ps @ _*) => MergeStrategy.discard
    case x => MergeStrategy.first
  }
}

