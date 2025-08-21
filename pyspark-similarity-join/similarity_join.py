import sys
import math
from pyspark import SparkConf, SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import udf, explode, col, count, collect_list, collect_set, row_number
from pyspark.sql.types import *
from pyspark.sql.window import Window

class Project3:
    def run(self, input_path, output_path, k):
        """Executes the main logic of the program."""
        # Convert k to float
        self.k = float(k)
        # Initialize Spark session
        spark = self.initialize_spark_session()
        # Read and prepare data
        transactions_df = self.read_and_prepare_data(spark, input_path)
        # Calculate item frequencies
        frequency_df = self.calculate_frequencies(transactions_df, spark)
        # Calculate similar items
        similarity_df = self.calculate_similar_items(transactions_df, frequency_df)
        # Write output
        self.write_output(similarity_df, output_path)
        # Stop Spark session
        spark.stop()

    def initialize_spark_session(self):
        """Initializes and returns a Spark session."""
        # Configure Spark settings
        conf = SparkConf()
        conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        conf.set("spark.executor.memory", "4g")
        conf.set("spark.executor.cores", "4")
        conf.set("spark.driver.memory", "2g")
        sc = SparkContext(conf=conf)
        return SparkSession.builder.master("local").appName("Project3_DF").getOrCreate()

    def read_and_prepare_data(self, spark, input_path):
        """Reads and prepares the data from the given input path."""
        # Define schema for input data
        schema = StructType([
            StructField("RecordId", StringType(), True),
            StructField("Name", StringType(), True),
            StructField("Quantity", IntegerType(), True),
            StructField("Date", StringType(), True),
            StructField("Price", DoubleType(), True)
        ])

        # Read input data
        data_df = spark.read.csv(input_path, schema=schema)
        data_df = data_df.drop("Quantity", "Price")

        # Parse year and cast types
        parse_year_udf = udf(self.parse_year, StringType())
        data_df = data_df.withColumn("Year", parse_year_udf(data_df.Date).cast("int"))
        data_df = data_df.withColumn("RecordId", data_df.RecordId.cast("int"))

        # Aggregate transactions
        transactions_df = data_df.groupBy("Year", "RecordId").agg(collect_set("Name").alias("Items"))
        transactions_df.cache()

        return transactions_df

    def calculate_frequencies(self, transaction_df, spark):
        """Calculates and returns the frequencies of items in transactions."""
        # Explode items and count frequencies
        frequency_df = transaction_df.select(explode(transaction_df.Items).alias("Item"))
        frequency_df = frequency_df.groupBy("Item").agg(count("*").alias("count"))
        frequency_df = frequency_df.select("Item", "count")

        # Create a ranking window
        window = Window.orderBy(frequency_df["count"])
        frequency_df = frequency_df.withColumn("pos", row_number().over(window))

        # Broadcast item frequency
        item_count = {row['Item']: row['pos'] for row in frequency_df.collect()}
        self.item_count = spark.sparkContext.broadcast(item_count)

        return frequency_df

    def calculate_similar_items(self, transaction_df, frequency_df):
        """Calculates and returns similar items based on Jaccard similarity."""
        # Vectorization and prefix length calculation
        vectorized_df = self.vectorize_transactions(transaction_df)
        prefixed_df = self.calculate_prefixes(vectorized_df)

        # Explode pre_items for similarity calculation
        exploded_df = self.explode_pre_items(prefixed_df)

        # Calculate similarity
        similarity_df = self.compute_similarity(exploded_df)

        return similarity_df

    def vectorize_transactions(self, df):
        """Applies vectorization to transaction items."""
        return df.withColumn("item_vec", udf(self.vectorize_items, ArrayType(IntegerType()))('Items', 'RecordId', 'Year'))

    def calculate_prefixes(self, df):
        """Calculates prefixes for vectorized transactions."""
        df = df.withColumn("prefix", udf(self.calculate_prefix_length, IntegerType())('item_vec'))
        return df.withColumn("pre_items", udf(self.get_prefix_items, ArrayType(StructType(
            [StructField("index", IntegerType(), nullable=False),
            StructField("cur_prefix", IntegerType(), nullable=False)
            ])))('item_vec', 'prefix'))

    def explode_pre_items(self, df):
        """Explodes pre_items for further processing."""
        df = df.withColumn("pre_item", explode(df.pre_items)).select("pre_item", "item_vec")
        return df.select(col("pre_item.index"), col("pre_item.cur_prefix"), col("item_vec"))

    def compute_similarity(self, df):
        """Computes similarity based on pre_items."""
        similar_udf = udf(self.calculate_similarity, ArrayType(StructType(
            [
                StructField("term1", IntegerType(), nullable=False),
                StructField("term2", IntegerType(), nullable=False),
                StructField("sim", DoubleType(), nullable=False),
            ]
        )))
        df = df.withColumn("vec", udf(self.update_vector, ArrayType(IntegerType()))("index", "item_vec"))
        df = df.repartition("cur_prefix")
        df = df.groupBy("cur_prefix").agg(similar_udf(collect_list("vec")).alias("similars"))
        df = df.withColumn("similar", explode(df.similars))
        return df.select(col("similar.term1"), col("similar.term2"), col("similar.sim")).dropDuplicates(["term1", "term2"]).orderBy(col("term1"), col("term2"))


    def write_output(self, df, output_path):
        """Writes the output to the specified path."""
        formatted_df = df.withColumn("formatted_output", udf(self.format_output, StringType())("term1", "term2", "sim")).select("formatted_output")
        formatted_df.write.format('text').save(output_path)

    def parse_year(self, date_string):
        """Extracts year from the date string."""
        date = date_string.split(" ")[0]
        year = date.split("/")[2]
        return str(year)

    def vectorize_items(self, items, record_id, year):
        """Vectorizes items for a given record."""
        position_list = [self.item_count.value[item] for item in items]
        return [year, record_id] + sorted(position_list)

    def calculate_prefix_length(self, items):
        """Calculates the prefix length for Jaccard similarity computation."""
        items_count = len(items) - 2
        return items_count - math.ceil(self.k * items_count) + 1

    def get_prefix_items(self, item_vector, prefix):
        """Extracts prefix items from the item vector."""
        return [[index, item_vector[index + 2]] for index in range(prefix)]

    def calculate_similarity(self, vector_list):
        """Calculates Jaccard similarity between pairs of item vectors."""
        result = []
        for i in range(len(vector_list)):
            for j in range(i + 1, len(vector_list)):
                if not self.filter_transaction_pairs(vector_list[i], vector_list[j]):
                    continue
                sim = self.jaccard_similarity(vector_list[i], vector_list[j])
                if sim >= self.k:
                    result.append([min(vector_list[i][2], vector_list[j][2]), max(vector_list[i][2], vector_list[j][2]), sim])
        return result

    def update_vector(self, index, vector):
        """Updates the vector with the given index."""
        vector.insert(0, index)
        return vector

    def format_output(self, term1, term2, sim):
        """Formats the similarity output."""
        return f'({term1},{term2}):{sim}'

    def filter_transaction_pairs(self, vec1, vec2):
        """Filters transaction pairs based on conditions."""
        if vec1[1] == vec2[1] or vec1[2] == vec2[2]:
            return False
        len1, len2 = len(vec1) - 3, len(vec2) - 3
        if self.k * len1 > len2 or self.k * len2 > len1:
            return False
        return min(len1 - vec1[0], len2 - vec2[0]) >= self.adjusted_ceil((self.k / (1 + self.k)) * (len1 + len2))

    def jaccard_similarity(self, vec1, vec2):
        """Computes the Jaccard similarity between two vectors."""
        set1, set2 = set(vec1[3:]), set(vec2[3:])
        return len(set1 & set2) / len(set1 | set2)

    def adjusted_ceil(self, x):
        """Adjusts the ceiling function to account for floating-point precision."""
        epsilon = 1e-9
        return round(x) if abs(x - round(x)) < epsilon else math.ceil(x)

if __name__ == '__main__':
    Project3().run(sys.argv[1], sys.argv[2], sys.argv[3])
