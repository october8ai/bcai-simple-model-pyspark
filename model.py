from pyspark.sql import SparkSession
from pyspark.sql.functions import col,lit
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.classification import LogisticRegression
from pyspark.mllib.util import MLUtils


spark = SparkSession.builder.master("local[4]").getOrCreate()

events = (spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("sample_data/events.csv"))

conversions = (spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("sample_data/conversions.csv")
    .withColumn("CONVERTED", lit(1.0)))


labeled_events = events.join(conversions, "ID", "left")

labeled_events.printSchema()    
labeled_events.show()


as_double = (lambda v: float(v) if v is not None else 0.0)

featureData = (labeled_events.rdd.map(lambda r: LabeledPoint(
          as_double(r["CONVERTED"]),
          Vectors.dense(
              as_double(r["HOTEL_CITY_ID"]),
              as_double(r["TIME_TO_ARRIVAL"]), 
              as_double(r["TIME_SPENT_ON_SITE"]))))).toDF()

training_and_test_data = MLUtils.convertVectorColumnsToML(featureData).randomSplit([0.7, 0.3])

lr = LogisticRegression(probabilityCol = "probability")

model = lr.fit(training_and_test_data[0])

auc = model.summary.areaUnderROC
print('Training AUC: ' + str(auc))

test = model.transform(training_and_test_data[1])
test.select("features", "label", "probability").show()




spark.stop()