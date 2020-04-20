The libraries used are:
- numpy
- pandas
- matplotlib
- sklearn
- seaborn
- scipy
- json
- glob
- pickle
- nltk
- gensim

RE_Notebook1

I started with data exploration and visualisation, checked how balanced different categories are.
Then changed format of different columns to make understanding of data easier, tried clustering using product_category_tree column to understand the dataset.

Then tried to estimate the similarity between different products using product description, to recommend similar products. Extracted the content related to each product. Made a bag of words, from the extracted content. Applied tf-idf to give preference to important words. Then calculated the similarity between different products. Sorted the list, and recommended the most similar 5 products as recommendations. We made recommendations for furniture, shoes, and Automative products as examples.

The Notebook is documented at almost every-step.