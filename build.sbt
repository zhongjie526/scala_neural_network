name := "stock_deep_learning"
version := "1.0"
scalaVersion := "2.11.11"
mainClass := Some("com.frank.AnnApp")

libraryDependencies ++= Seq(
"org.scalanlp" %% "breeze" % "1.0-RC2",
"org.scalanlp" %% "breeze-natives" % "1.0-RC2",
"com.opencsv" % "opencsv" % "4.0"
)


