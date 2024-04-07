# Closing-Price-Prediction
#ABSTRACT

##Modeling Approach

For this , I employed a deep learning approach using Tensorflow and Keras to predict stock prices for six different companies over the next 96 days. Here’s an overview of our modeling approach:
##Modeling Approach:
Data Preparation: Gather historical stock price data and preprocess it by handling missing values, normalizing the data, and engineering relevant features.
Model Architecture Selection: Choose a Sequential model architecture and configure it with GRU layers for capturing temporal dependencies in the data.
Model Compilation: Compile the model using mean squared error loss and the Adam optimizer to train the model effectively.
Training: Train the model on the preprocessed data, fine-tuning hyperparameters as necessary.
Evaluation: Evaluate the model's performance using metrics such as mean absolute error, mean squared error, and visual inspection of predicted versus actual stock prices.

##Model Architecture:

 This architecture comprises three GRU layers with 32 units each, the first two layers returning sequences. A dropout layer with a dropout rate of 20% is added to prevent overfitting. Finally, a Dense layer with a single unit is used for output, and the model is compiled using mean squared error loss and the Adam optimizer.
Data Preparation and Preprocessing
Univariate Strategy:
Our strategy centered on a univariate time series approach specifically tailored for predicting closing prices. Despite considering multivariate methods, we consistently observed superior performance with the univariate approach, particularly in terms of prediction accuracy.
Normalization Using MinMaxScaler:
We adopted MinMaxScaler to scale down the closing prices within a range from 0 to 1. This normalization technique was instrumental in ensuring numerical stability and facilitating convergence during training. By mitigating the influence of outliers and addressing discrepancies in scales across various stocks, it significantly enhanced the robustness of our models.

##PREDICTIONS

Utilizing the Sliding Window Method:
For training, our model was primed with a look-back window of 200 timesteps, where it analyzed 200 consecutive historical data points to forecast the subsequent day's closing price.
During the prediction phase for the test dataset, we adopted a recursive approach. Initially, we utilized the latest 200 values from the training set to predict the first value of the test set. Subsequently, for each prediction, we employed the most recent 199 values from the training dataset along with the previously predicted value. This ensemble of 200
1

time steps was then iteratively fed into the model to generate successive predictions.

##RESULTS

Peak Performance:
After multiple submissions and refinement cycles, our model achieved its peak performance, yielding an RMSE (Root Mean Square Error) of 111.8159 across 52% of the test dataset. This metric represents the average difference between actual and predicted closing prices.
Performance Fluctuations:
It's noteworthy that our model exhibited fluctuations in performance across different segments of the test dataset. While it maintained an RMSE of 111.8159 on 52% of the test data, its efficacy declined to 345.5134 in other segments.
Additionally, it's important to highlight that one of our model initially obtained an RMSE of 244.8535 on 52% of the dataset, but notably improved to 110.819. This variability underscores the sensitivity of our model's performance to various data subsets and highlights the iterative nature of model refinement.

##OTHER APPROACHES

Here are the other models and approaches that we used:
1. LSTM
2. Transformers
3. Linear Regression 4. Multivariate RNN
