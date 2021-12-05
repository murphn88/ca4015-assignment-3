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


Recommender systems are extremely prevelant and can vary in levels of complexity. Within the project alone, I created four different models, some with different variations. I initally followed a Google Colab tutorial to make a matrix factorization recommender system. Within that, I experimented with different methods including regularlization and similarity measures. I then used an open source library to assist in making a Light Graph Convolution Network recommender, which resulted in highly relevant recommendations, even without parameter tuning. Finally, I made a simple content-based recommender system, which can identify two Italian opera tenor as similar based soley on the tags that they both received. While the content-based model was shown not to be suitable on its own, it could be combined with another system to improve performance.

One of my key takeaways from this project is the importance of data pre-processing. I initally attempted to create the matrix factorization model without reseting the indices and making them consecutive. This was problamatic to tensorflow and due to the size of the model, debugging took quite a bit of time. Additionally, the decision on how to deal with the weight (listen counts) feature was higly impactful for the rest of the work. Due to the computational complexity of most machine learning models if pre-processing is not done correctly a lot of time can be wasted going back and processing the data properly before retraining the model. This project has highlighted importance good pre-processing practices.

A proposal for future work is to make a hybrid collaborative filtering/ content-based model. This would perform collaborative filtering to produce a set of potential recommendations. The content-based model would then rank the potential recommendations to arrive at the optimal recommendations. Another area for further work is focusing on the relationship data betweeen friends. Due to time constraints, I neglected to use this data. I would like to further examine the similarity between friends, particularly whether they share an interest in certain, potentially niche, genres and whether harvesting these relationships can improve recommendations.
<!-- #endregion -->
