# Project_Portfolio
My Data Science Project Portfolio.
Model Training
As mentioned previously in section 3.2, the second deliverable for our project involves more predictive analytics and machine learning, used to forecast ISS regions. Before conducting our analysis, we preprocessed our dataset to choose relevant attributes, target variables, and make sure valid entries were included. Since we are working with time dependent data, we tried multiple time series models. For time series forecasting we implemented the Autoregressive Integrated Moving Average (ARIMA) model and the Long Short-Term Memory (LSTM) model. The dataset we focused our training and testing on was on the Buffalo, New York IGRC Dataset. 

Before training and testing both the models, we needed to preprocess the dataset to have the proper variables to feed into our model. We first filtered the dataset to the years 2022 and 2023, which helped to reduce the model complexity. Additionally, reducing the dataset led to more consistent data recordings from the radiosonde. Years too far back would result in much more missing data than newer years. Lastly, as for deciding which years to pick, this would be a research gap that would need to be looked into more into the future as determining the best length of time for our time series models would need to be simulated(medium). A second preprocessing task we focused on was converting the ISSC column, which was represented as a character data type, into an integer, which will allow us to feed into our model. The third preprocessing task was to make the date column the index as this was needed for our time series models. Lastly, upon some exploratory data analysis, we removed extreme outliers. We had one day that experienced 55 instances of ice supersaturated conditions, which in our initial testing threw off the models as the mean of times per day that ice supersaturated conditions were present was around 2. 

After preprocessing, we went on to feature engineering. We created an ISSC volume feature which represented the number of instances per day ice supersaturated conditions occurred and that would be our dependent variable that our models would try to predict. As for our independent variables, we tested the following:
1.	Average Daily Temperature – The mean temperature per day.
2.	Average Daily Relative Humidity to Ice – The mean relative humidity to ice per day.
3.	Rolling Average – The average value of ice supersaturated condition volume per day within a 5-day window size.
4.	Exponential Smoothing – The average value of ice supersaturated condition volume per day within a 5-day window size, assigning weights to each observation. As the observation becomes older, it gets assigned a lower weight.
5.	Temperature Daily volume – The number of times per day the temperature was recorded as below -42 degrees Fahrenheit. 
6.	Relative Humidity to Ice Daily Volume – The number of times per day that relative humidity to ice was above 100 percent. 
7.	Lag Features – Taking the past values and incorporating them into the current prediction.
 
We initially started working with the ARIMA model. We built 

After some unsuccessful attempts we switched to using an LSTM model. Initially we started just by using lag variables of ‘volume’ (how many ISSCs the day had) for the predictions. We progressed through using other attributes like rolling averages and exponential smoothing of the ‘volume’ attribute, as well as averages of temperature and relative humidity to ice. This improved our predictions slightly, but we got the best results when adding ‘volume of temp’ and ‘volume of rhi’ (These variables are counts of how many times temperature and rhi were ISSC causing each day). To make sure we would get the best results we first determined how many lag variables to include and for what attributes. We then implemented another grid search to find the optimal hyper parameters such as learning rates and batch size. To visualize this better we created a heat map of the LSTM parameters, showcasing which combinations had the lower Mean Squared Error.

Model Evaluation
Model Validation

DAEN 690 Presentation April 2, 2024
Slide 66 – Machine Learning – ISSC Volume Predictions
•	A big focus of our project has been on the data visualization side for analysis of past ice supersaturated conditions which can be helpful for climatologists and meteorologists. However, secondarily, our client wanted a predictions dashboard, which meant creating some type of machine learning model that could potentially predict if an ice supersaturated region would form on a given day, which would allow airline companies to adjust their flight plans accordingly to avoid these ice supersaturated regions. 
•	To achieve this, we scoped our goal to predict the volume of ISSC for the next calendar day, using a time series model to forecast these future days. A day with higher volume levels would indicate a day that would likely experience ice supersaturated conditions and if the volume is high enough, could explain the vertical depth and number of hours the conditions would occur. 
•	Our end goal is after the modeling is completed, to integrate the prediction model into the predictive dashboard. 
•	In these next couple slides I’m going to go over the two machine learning models we built and evaluated, which were the Autoregressive Integrated Moving Average model and the Long Short-Term memory model
Slide 67 - Machine Learning – Preprocessing/Feature Engineering
•	Before going into the two models, I want to first talk about the various preprocessing and feature engineering aspects that we did.
•	For preprocessing we 
o	Filtered the data to the years 2022 and 2023, which helped to reduce the model complexity as well as more recent information could be a better predictor for future events as our time prediction window is shorter than long term.
o	Converted the ISSC column character to integers to have a proper dependent variable to feed into our model
o	Made the date the index as this was needed for our time series models
o	Removed extreme outliers. We had one day that experienced 55 instances of ice supersaturated conditions, which in initial testing really threw off our models as the mean was around 2 instances for days with ISSC.
•	For Feature Engineering we created and tried a variety of different features in our model
o	Originally, we created a volume feature which basically calculated the number of instances per day ice supersaturated conditions occurred and that would be our dependent variable. 
o	Our independent variables we created after that were lag variables for volume of ISSC per day.
o	Then we tried the mean daily temperature and relative humidity to ice, which improved our model predictions more
o	From there we thought maybe try creating an improved version of those two features and created a temperature daily volume feature which represents the number of times per day the temperature is below -42 degrees Fahrenheit and the relative humidity to ice daily volume feature which represents the number of times per day the relative humidity to ice is above 100 percent. 
o	We tried other features like rolling average and exponential smoothing but had little success with those features.
Slide 68 – Machine Learning – ARIMA Model Evaluation
•	After doing some feature engineering, we evaluated the ARIMA model and LSTM model. 
•	For the ARIMA model, we first built the model and then created a parameter tuning function which basically tried various combinations of P, D, and Q values.
•	P represents the number of lag observations in the model
•	D represents the number of times the raw observations are differenced
•	Q represents the size of the moving average window.
•	When running our parameter tuning function, we were given 5, 0, 2 as our optimal parameters.
•	When plugging these parameters into our ARIMA model, we were left with a mean square average of 0.988 and the following Actual versus predicted plot, which as you can see is underwhelming.
Slide 69 – Machine Learning – LSTM Model Evaluation
•	Moving on to the LSTM model, in our initial testing we found better success and so more of the effort was taken on optimizing this model. 
•	To fully optimize this model, we first evaluated the optimal number of lag variables for ISSC Daily Volume, Temperature Daily Volume, and Relative Humidity to Ice Daily Volume.
•	After creating a function to run the different combinations, we found that ISSC Daily Volume of 2, Temperature Daily Volume of 0, and Relative Humidity to Ice Daily Volume of 1 was the optimal number of lag features for each of our variables.
•	After optimizing the lag variables, we went to optimizing the hyperparameters. 
•	We tested different values for units, learning rates, and batch sizes.
•	We visualized the different combinations in a heatmap shown on the right-hand side.
•	The optimal hyperparameters were units set to 100, learning rate set to 0.01 and batch size set to 64. 
Slide 70 – Machine Learning – LSTM Model Evaluation
•	As a result of the lag variable testing and hyperparameter tuning, we trained and tested a newly developed model that was more fully optimized. As shown at the bottom, the left-hand side was our model’s performance before the optimization and on the right is our optimized model. 
•	Although we saw a great improvement, limitations persist that will need be researched in future projects, such as 
o	Addressing the missing values from an incomplete radiosonde dataset
o	New research into understanding the unpredictability of relative humidity to ice
o	And additional modeling time with additional computing resources to dive even deeper into the feature engineering and parameter tuning.
Slide 71 – Risks and Planned Mitigations
•	As for risks and mitigations, very similar issues still persist with trying to keep all programmers on the same page as well as keeping communication channels open with the Tableau developers.
•	Additionally, as we are nearing the end of the semester, trying to keep on top of the timeline of the project. 
Slide 72 – Weekly Partner Meeting
•	This week we will be meeting with Dr. Sherry this Thursday. We will be going over our Tableau dashboard as well as go over the modeling results and get his thoughts on both before heading into our last Sprint. 

