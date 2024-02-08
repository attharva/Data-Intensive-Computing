Diving into New York’s AIRBNB: More than just numbers

PROJECT REPORT (CSE 587)
Anuj Vadecha | 
Sowmya Iyer |
Atharva Kulkarni 

Introduction
PART 1 : PROBLEM STATEMENT
Background
NYC has always been on our bucket list. The city's rhythm, its people, the skyscrapers, and of
course, the pizza. Every year, millions flock to the city, and many, like me, prefer Airbnb to
traditional hotels. Given the city's magnetic pull and Airbnb's growing clout, I wanted to
understand how the two dance together. Plus, for anyone thinking of hosting their place on
Airbnb, wouldn't it be cool to know how to set the right price or what guests really care about?

Problem
Every time I've traveled, I've been faced with a common question: hotel or Airbnb? And more
often than not, I've leaned towards Airbnb for that authentic, local experience. But what drives
the prices on Airbnb? Why are some places more popular than others? And how does the vibe
of a neighborhood influence an Airbnb listing? With these personal curiosities in mind, I decided
to delve deep into Airbnb's listings in New York City. I aim to uncover: The price game: What's
the deal with varying prices across different room types and boroughs? The neighborhood stars:
Why are some neighborhoods buzzing with listings while others aren’t? Reviews & Pricing: Is
there a link between how much a place costs and what people say about it?

Significance
Understanding the Airbnb scene in NYC isn't just about data. It's about stories, experiences,
and the dreams of travellers. But it's also about:
Hosts: Knowing how to make their space stand out and maybe earn a bit more. Traveler's:
Figuring out where to get the best bang for their buck. City Planners: Realizing how Airbnb
shapes neighborhoods and the city's pulse. Aspiring Entrepreneurs: Spotting gaps in the market
and coming up with the next big idea. b. Potential Contribution: Through this project, I hope to
shed light on the stories behind the numbers. By diving into NYC's Airbnb data, I want to craft a
narrative that can guide hosts, help travelers, inspire entrepreneurs, and maybe even catch the
eye of city planners.


Why This Matters
Imagine a host in Brooklyn realizing they could earn more just by tweaking their listing a bit. Or a
traveler finding a hidden gem of a neighborhood they'd never considered before. Or even an
entrepreneur spotting a trend and starting something new. That's the power of data. And
through this project, I hope to harness it, share it, and make a difference.

PART 2: DATA SOURCES
Sources
We will be using the New York city airbnb data
• Link: - https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data/
• Credits - DGOMONOV

Working Instructions for the Streamlit App

Introduction
This report details the development and functionality of a Streamlit application designed to analyze and visualize New York Airbnb data. 
The application leverages Decision Tree and K- means algorithms to provide insights into the Airbnb market in New York.
System Requirements
Python environment (any version).
A trained Random Forest Regressor model saved as ‘airbnb_price_prediction_model.joblib'. 
There are a few requirements that are specified in the requirements.txt that need to be installed in your python virtual environment for the set up. 
Following are the steps to setup the virtual environment.

Environment Setup

Set up a Python virtual environment
python -m venv venv

Activate the virtual environment
macOS/Linux: source venv/bin/activate

Installing Streamlit and Dependencies
These dependencies are follows. To install these in your python environment, you can run this command.
pip install -requirements.txt

Setting up the Random Forest Regressor model
To initiate the Random Forest Regressor model, execute the following command:
python rfregression.py

Please be patient during this process, as running the model might take some time due to its
computational intensity. This will save the model automatically as airbnb_price_prediction_model.joblib and you can use this while running the app for your predictions.

Launch the application
Launch the app by running the following command in the terminal:
streamlit run app_name.py

Navigating the Interface
The app interface includes

Data Upload Section To upload the Airbnb dataset.
The app requires the New York Airbnb dataset in CSV or TXT format. Ensure the dataset
includes columns for price, number of reviews, neighborhood, and room type. You can use the
drag-and-drop feature in the data upload section to upload the Airbnb dataset.

Sidebar For user inputs and controls.
The sidebar in the app provides various controls that allow users to filter and analyze the data
according to their preferences:
• Minimum Price This slider sets the lower limit for the price range of the listings to be included
in the analysis.
• Maximum Price This slider sets the upper limit for the price range.
• Minimum Reviews Users can set a threshold for the minimum number of reviews a listing
should have to be included. This helps in focusing on more popular or frequently reviewed
listings.
• Minimum Nights This option allows users to filter listings based on the minimum number of
nights required for a stay.
• Neighborhood Groups Users can select or deselect any of the five NYC islands (Manhattan,
Brooklyn, Queens, The Bronx, Staten Island) to include or exclude them from the analysis.
This feature allows for a more targeted examination of specific areas.
These controls are interactive and the visualizations and analyses in the app will update in real-
time based on the user’s selections and inputs. This dynamic feature enables users to explore
and understand the data from multiple perspectives, providing a comprehensive overview of the
New York Airbnb market.

Part 3
EXPLORATORY ANALYSIS
MODELS FOR CLUSTERING AND PREDICTION

