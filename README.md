## AI-Symptom-Checker
#**Overview**
  The AI-Powered Symptom Checker is a web application designed to analyze user-reported symptoms and provide potential disease predictions along with recommendations for further action. By leveraging K-    Nearest Neighbors (KNN) machine learning algorithms, this tool aims to enhance health literacy, improve user experience, and promote timely healthcare interventions.

**Demonstration video:** https://youtu.be/W9iGcD9pTzU?si=Vrtj20tWelxv2z9s

https://github.com/user-attachments/assets/3616c604-bb62-48b4-a326-8d19e1f23e2d
  
**PPT Link:**
https://mujmanipal-my.sharepoint.com/:p:/g/personal/kanak_23fe10cse00232_muj_manipal_edu/EUVmqZHuWjhMsgUBYEj5ZKoB_HEgNXKp5fxr8ZdwuaikRg?e=ZvgzEY

**Drive link:**
https://drive.google.com/drive/folders/1ZXyIEl3cSG9AtZdsHX-0KxA__xFd3jzn

#**Features:**

  -Symptom Analysis: Users can input symptoms to receive a preliminary analysis and potential disease predictions.
  
  -Predictive Analytics: The system forecasts potential health issues based on historical trends in user symptoms and behaviors.
  
  -User-Friendly Interface: A simple and intuitive web interface allows users to interact with the symptom checker easily.
  
  -Real-Time Monitoring: Users can track their symptoms over time and receive alerts for concerning trends.
  
  -Educational Resources: Provides information about symptoms and associated health conditions to improve health literacy.

#**Technologies Used:**

**Programming Languages:**

  -Python: Backend logic and machine learning model implementation.
  
  -JavaScript: Front-end interactivity.
  
  -HTML/CSS: User interface development.

**Framework:**

  -Flask: Web framework for managing backend and APIs.

**Machine Learning:**

  -Scikit-learn: Library used for implementing the KNN algorithm.

**Data Storage:**

  -Cloudant: NoSQL database for storing user consultations and symptom data.

**Libraries:**

  -NumPy: For numerical operations and data manipulation.
  
  -Pandas: For data handling and preprocessing.
  
  -Joblib: For saving and loading the trained model.

#**Installation:**

To set up the project locally, follow these steps:

-Clone the repository:

  '''bash
  Copy code
  git clone <repository-url>
  cd symptom-checker
  Install the required packages:

  '''bash
  Copy code
  pip install -r requirements.txt
  Set up your Cloudant credentials in the environment variables or directly in the code.

-Run the application:

  '''bash
  Copy code
  python app.py
  Access the application at http://127.0.0.1:5000.

-Usage

  Input symptoms in the provided field on the main page.
  
  Click on "Check Disease" to receive predictions and recommendations.

#License:
This project is licensed under the MIT License. See the LICENSE file for details
