import pandas as pd
import time
from mlxtend.frequent_patterns import fpgrowth, association_rules


class FPGrowthMiner:
    def __init__(self):
        pass

    def run(
        self,
        basket_bool: pd.DataFrame,
        min_support: float = 0.01,
        min_confidence: float = 0.3,
        min_lift: float = 1.0,
    ):
        """
        Chạy FP-Growth và sinh luật kết hợp
        """

        start_time = time.time()

        # 1. Khai thác tập phổ biến
        frequent_itemsets = fpgrowth(
            basket_bool,
            min_support=min_support,
            use_colnames=True
        )

        # 2. Sinh luật kết hợp
        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=min_confidence
        )

        # 3. Lọc theo lift
        rules = rules[rules["lift"] >= min_lift]

        execution_time = time.time() - start_time

        print(f"FP-Growth finished in {execution_time:.2f} seconds")
        print(f"Frequent itemsets: {len(frequent_itemsets)}")
        print(f"Association rules: {len(rules)}")

        return frequent_itemsets, rules, execution_time
