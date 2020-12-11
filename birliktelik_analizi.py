# Kütüphaneler eklendi.

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Veri setimizi okuttuk.
df = pd.read_csv("data/GroceryStoreDataSet.csv", header=None)

df.head()

df.shape

# Veri setimizin herhangi bir değişken ismi yok, bu yüzden product değişken oluşturduk.
df.columns = ["Products"]

# Verilerimiz birleşik olduğundann bunları "," le göre ayırma işlemi yapacağız.
# Bunun için isimsiz fonk. kullandık. Product değişkeninin yakalanan her elemanının içinde ","'e göre ayırma işlemi yaptık
pure_df = df.Products.apply(lambda x: x.split(",")).copy()

# tipine bakıyoruz.
type(pure_df)

# Burada değişkenlerimizi 0-1 şeklinde ayarladık. 1 alınma, 0 alınmama durumu
temp = TransactionEncoder()
temp_df = temp.fit(pure_df).transform(pure_df)
new_df = pd.DataFrame(temp_df, columns=temp.columns_)
new_df.head()


# Support değerimizi %20 olarak belirledik
supps = apriori(new_df, min_support=0.20, use_colnames=True)
supps


# oluşan verimizi supportiçin  azalan değerlere göre sıraladık
supps.sort_values(by = "support", ascending = False)

conf = association_rules(supps, metric="confidence", min_threshold = 0.15)
conf.sort_values(by ="confidence", ascending = False)
