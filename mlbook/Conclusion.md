---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region -->
# Conclusion


Recommender systems are extremely prevelant and can vary in levels of complexity. Within the project alone, I created four different models, some with different variations. I initally followed a Google Colab tuturial to make a matrix factorization recommender system. Within that, I experimented with different methods including regularlization and similarity measures. I then used an open source library to assist in making a Light Graph Convolution Network recommender, which resulted in even better results, even parameter tuning. Finally, I made a simple content-based recommender system that is based soley on artists' tags. While this model is not suitable to be used on its own, it could certainly be combined with another system to improve performance.

One of my key takeaways from this project is the importance of data pre-processing. I initally attempted to create the matrix factorization model without reseting the indices and making them consecutive. This was problamatic to tensorflow and due to the size of the model, debugging took time as the error messages often did not point to the root cause. Additionally, the decision on how to deal with the weight (listen counts) feature higly impacted the rest of the work. This has highlighted importance good pre-processing practices.


A proposal for future work is to make a hybrid collaborative filtering/ content-based model. This would perform collaborative filtering to produce a set of potential recommendations and the content-based model would then rank the potential recommendations. Prehaps with a different dataset that contains more content information. Another area for further work is including the relationship betweeen. Due to time constraints, I neglected to use the friends data, but if I had more time I would have liked to have delved further into the similarity between friends, particularly whether they have a certain niche genre that unites them.
<!-- #endregion -->
