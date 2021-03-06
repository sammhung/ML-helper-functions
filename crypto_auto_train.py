def time_series_ai(data_dir, 
                          crypto_name,
                          custom_model="n/a",
                          date_column="Date",
                          price_column="Open",
                          view_crypto_history_graph=False,
                          graph_size=(10, 7),
                          test_split=0.2,
                          use_custom_model=False,
                          plot_model_training=False):
  
  """
  This function will create, train and evaluate a deep neural network to predict the price of stocks or crypto.

  Args: 
    data_dir : Directory of Crypto/Stock CSV file
    crypto_name : Name of the Crypto/Stock
    custom_model : Use a custom model architecture for predicting crypto/Stock prices (default = n/a)
    date_column : Name for the Date column (default = Date)
    price_column : Name for the Price column (default = Open)
    view_crypto_history_graph : View the historic graph for the crypto (default = False)
    graph_size : Size of all graph (default = (10, 7))
    test_split : Split size for evaluating the model (default = 0.2)
    use_custom_model : Must be set to True if you want to use your own model (default = False)
    plot_model_training : Plot the training of the model (default = False)

  Returns: 
    Model, Model Results, Predict Price, Naive Model results
  """

  from timeseries_helper import make_windows, make_train_test_splits, evaluate_preds, plot_time_series
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  import tensorflow as tf
  
  # Part 1: Reading in, processing and understanding Data

  # Reading in crypto data
  crypto_data = pd.read_csv(data_dir).dropna()

  # Get timestamps
  if(date_column != "n/a" or "N/A"):
    timesteps = pd.DatetimeIndex(crypto_data[date_column])

  # Get prices
  prices = np.array(crypto_data[price_column])

  # Plot Crypto historic graph
  if view_crypto_history_graph:
    plt.figure(figsize=graph_size)
    plt.plot(timesteps, prices)
    plt.title(crypto_name)
    plt.xlabel("Year")
    plt.ylabel("Price")


  # Part 2: Splitting the data

  # Creating windows and labels
  windows, labels = make_windows(prices)

  # Splitting up the data
  train_windows, test_windows, train_labels, test_labels = make_train_test_splits(windows, 
                                                                                  labels, 
                                                                                  test_split=test_split)
  # Expanding dimensions so it'll work with model
  train_windows = tf.expand_dims(train_windows, axis=1)
  test_windows = tf.expand_dims(test_windows, axis=1)

  # Creating a save best model callback
  def model_checkpoint():
    return tf.keras.callbacks.ModelCheckpoint("model_checkpoints/model",
                                              save_best_only=True,
                                              verbose=0)
  
  # Part 3: Creating and fitting the model
  if(use_custom_model):
    model = custom_model

  else:
    model = tf.keras.Sequential([
      tf.keras.layers.RNN(tf.keras.layers.LSTMCell(32, activation="relu")),
      tf.keras.layers.Dense(128, activation="relu"),
      tf.keras.layers.Dense(1, activation="linear")
    ])

    # Compiling model
    model.compile(loss="mae",
                  optimizer="adam",
                  metrics=["mae", "mse"])
  
  # Fitting the model
  model_history = model.fit(train_windows,
                            train_labels,
                            epochs=100,
                            verbose=0,
                            validation_data=(test_windows,
                                             test_labels),
                            callbacks=[model_checkpoint()])
  
  # Plot model training
  if(plot_model_training):
    pd.DataFrame(model_history.history).plot()

  # Load in the best model
  model = tf.keras.models.load_model("model_checkpoints/model")


  # Part 4: Evaluating the model

  # Make predictions
  model_preds = tf.squeeze(model.predict(test_windows))

  # Evaluate predictions
  model_results = evaluate_preds(tf.squeeze(test_labels),
                                 model_preds)
  
  # Plot the accuracy of the model
  plt.figure(figsize=graph_size)
  plot_time_series(timesteps[-len(test_labels):],
                   test_labels,
                   label="True values",)
  plot_time_series(timesteps[-len(test_labels):],
                   model_preds,
                   label="Model Predictions",
                   format="-",)
  plt.ylabel(crypto_name)
  
  # Part 5: Make a Naive model to see if this model should be used
  naive_prices = tf.squeeze(test_labels)[1:]
  naive_actual_prices = tf.squeeze(test_labels)[:-1]

  naive_results = evaluate_preds(naive_actual_prices,
                                 naive_prices)
  
  if naive_results["mae"] < model_results["mae"]:
    evaluation = "Not the greatest AI, but it should do the job well enough. Look below for the average error to beat.\n\
    If it's close enough it should be fine."
  else:
    evaluation = "This AI could be used for predicting this crpyo as it's quite accurate."

  # A function to predict data
  def make_predictions(x, print_evaluation=True, return_evaluation=False):
    past_data = tf.reshape(x, shape=(1, 1, 7))
    model = tf.keras.models.load_model("model_checkpoints/model")
    prediction = np.round(tf.squeeze(model.predict(past_data)), decimals=2)

    if print_evaluation:
      print(f"The price prediction for the next 24-72 hours is: {prediction.numpy()}")

    if return_evaluation:
      return prediction.numpy()

  # Predict tomorrow's price
  tmr_pred = tf.squeeze(model.predict(tf.reshape(tf.squeeze(test_labels)[-7:], shape=(1, 1, 7))))

  # Evaluation
  print("-----------")
  print(f"Evaluation: {evaluation}")
  print("-----------")
  print(f"Average error to try beat: ${naive_results['mae']}")
  print(f"The average error for this AI: ${model_results['mae']}")
  print(f"----------")
  print(f"Using the data you gave me, the price prediction for the next 24-72 hours is ${tmr_pred}.")
  print("-----------")
  print("Thanks for supporting HUNG'S TECH!")

  return model, model_results, make_predictions, naive_results,

