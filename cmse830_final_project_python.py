# loading libraries and packages
import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

salary_3 = pd.read_csv("salary_complete.csv")

#Feature Variables - Age, Gender, Education Level, Years of Experience, Salary
X = salary_3.iloc[:, [0,1,2,4,5]].values

#setting the target variable as the Job Title
y = salary_3.iloc[:, 3].values

#splitting the data into training and testing sets with a 80:20 ratio
test_fraction = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, random_state= 42)

#Decision Tree Classifier
my_classifier_dt = DecisionTreeClassifier(criterion='gini', random_state=0)
my_model_dt = my_classifier_dt.fit(X_train, y_train)
dt_score = my_model_dt.score(X_test, y_test)

#Title
st.title("And the Journey Continues...")
st.subheader("Job Recommendation System")

#Tab Order and Names
tab1, tab2, tab3, tab4 = st.tabs(["Introduction", "Job Recommender", "Conclusion", "Bio"])

#Tab 1 - Introduction
with tab1:
    col11, col12 = st.columns(2)
    with col11:
        image2 = Image.open("JOB.png")
        st.image(image2)
    with col12:
        st.write("Ready to continue this journey? Continue? Yes! This app serves the purpose of continuing the journey we have started in the previous app.")
        st.write("If you have not already done so, please visit this app (linked below) first to start your journey and then return here to continue: https://salarymidtermprojectupdatepy-6j2kvajxpbddwvrx6avjzx.streamlit.app/")
    st.write("Now, let's continue the journey!")
    st.write("Get ready to discover the job that's not just a match, but a perfect fit for you! ")
    st.write("Say goodbye to job hunting stress and hello to a brighter, tailored future!") 
    st.write("Navigate to the next tab **Job Recommender** and tell us a little about yourself. With your input, this system will recommend a job for you and more.")
    st.write("**HAPPY JOB HUNTING!**")



#Tab 2 - Job Recommender
with tab2:
    #with st.sidebar:
        #st.header("Input Key")
        #st.write("Age: 0 - 100")
        #st.write("Gender: Male, Female, Other")
        #st.write("Education: High School, Bachelors, Masters, PhD")
        #st.write("Years of Experience: 0 - 30")
        #st.write("Salary: 0 - 300000")
    st.write("This is the Job Recommender System. This system will recommend a job title based on your input.")
    st.write("Please enter your information in the form below and click submit. ")
    with st.form(key='user_input'):
        age = st.number_input("AGE", 0, 100)
        gender = st.selectbox("GENDER", ["Male", "Female", "Other"])
        education = st.selectbox("EDUCATION", ["High School", "Bachelors", "Masters", "PhD"])
        experience = st.number_input("YEARS OF EXPERIENCE", 0, 30)
        salary = st.number_input("SALARY", 0, 300000)
        submit = st.form_submit_button('Submit')
    if submit:
        st.write("**You entered the following information:**")
        st.write("Age: ", age)
        st.write('Gender:', gender)
        st.write('Education Level:', education)
        st.write('Years of Experience:', experience)
        st.write('Salary:', salary)

        #Predicting the top job title based on the user input

        #Creating a dataframe of the user input
        user_input = {'Age':age, 'Gender': gender, 'Education Level': education, 'Years of Experience': experience, 'Salary': salary}
        user_input_df = pd.DataFrame(user_input, index=[0])

        #Replacing the categorical variables with numerical values
        user_input_df['Education Level'].replace(["High School", "Bachelors", "Masters", "PhD"], [0, 1, 2, 3], inplace=True)
        user_input_df['Gender'].replace(["Male", "Female", "Other"], [0, 1, 2], inplace=True)
        
        #Predicting the top job title based on the user input
        user_input_df['Job Title'] = my_model_dt.predict(user_input_df)
        job = user_input_df['Job Title'][0]

        st.subheader("**We recommend the following Job Title:**") 
        st.subheader(job)

        st.write("To learn more about your recommended job title, please search for it on the website linked below:")
        st.write("https://www.onetonline.org/")

        st.write("Additionally, please edit the input values to see what other job titles are recommended for you.")
        st.write("If satisfied with your recommended job title, please navigate to the next tab **Conclusion** to conclude your journey.")

with tab3:
    st.write("In conclusion, this app uses the Decision Tree Classifier model to recommend a job title based on your input.") 
    st.write("The model was trained using a 80/20 split where the feature variables were **Age**, **Gender**, **Education Level**, **Years of Experience**, and **Salary** and the target variable was **Job Title**.")
    st.write("The model performed with an accuracy of 89%")
    st.write("Furthermore, the creator of this app would like to implement an interactive facet plot that allows the user to see how they compare with others in their recommended job title.")
    col1, col2 = st.columns(2)
    with col1:
        st.write("As suggested in the previous app, please continue to discover your interests, research careers and career paths, connect and network with professionals in your field, and more.")
        st.write("Thank you for using this app. We hope you enjoyed your journey and found it helpful.")
        st.write("Please visit the Bio tab to learn more about the creator of this app.")
    
    with col2:
        image5 = Image.open("Safe_Journey_Quotes_Artboard-1.jpg")
        imagenew5 = image5.resize((600,400))
        st.image(image5)
    
    #st.write("If you have any questions or concerns, please contact Jasmine Harris at

with tab4: 
    col11, col12 = st.columns(2)
    with col11:
        image = Image.open("IMG_8945_Original.JPG")
        st.image(image)
    with col12:
        st.write("Jasmine Anita Harris is a graduate student at Michigan State University studying Data Science. Her anticipated graduation date is April 2024.")
        st.write("Originally from Tuscaloosa, Alabama, Jasmine is a grateful daughter of Regina Harris, granddaughter of Tempie and Johnnie Gardner and Frederick and Flendora Harris, and dog mom to Luna Harris.") 
        st.write("Jasmine recieved her Bachelors of Arts degree in Mathematics from Spelman College in 2022.")
        st.write("She lives by the scripture Luke 1:19, quoted 'For with God nothing shall be impossible.'")
    st.write("At Michigan State University, Jasmine is a graduate research assistant currently working on a project that involves the analysis of students' models in an Introduction to Biology course by using network analytic metrics and clustering methods.")
    st.write("Outside of academics, Jasmine enjoys spending time with family and friends, listening to music, and playing tennis.")
    st.write("Jasmine aspires to become a Data Scientist in a field of her interest and make an impact in the world by using her skills to help others.")
    st.write("To learn more about Jasmine, please visit her LinkedIn page linked below:")   
    st.write("www.linkedin.com/in/jasmineanitaharris")
        #job_compare = st.checkbox("Click here to see how you compare with others in your recommended job title")
        #if job_compare:
    
            #user_input_df['Education Level'].replace([0, 1, 2, 3], ["High School", "Bachelors", "Masters", "PhD"], inplace=True)
            #user_input_df["Gender"].replace([0, 1, 2], ["Male", "Female", "Other"], inplace=True)

            #combined_df = pd.concat([salary_3, user_input_df], ignore_index=True)
            #combined_df['Education Level'].replace([0, 1, 2, 3], ["High School", "Bachelors", "Masters", "PhD"], inplace=True)
            #combined_df["Gender"].replace([0, 1, 2], ["Male", "Female", "Other"], inplace=True)

            #combined_df
            #output_job = combined_df[combined_df["Job Title"] == job]
            #output_job 


        #Displaying the top 5 job titles based on the user input
        #st.write("**Here are the top 5 job titles based on your information:**")
        #user_input_df['job_title'] = my_model_dt.predict_proba(user_input_df)
        #user_input_df = user_input_df.sort_values(by=['job_title'], ascending=False)
        #user_input_df = user_input_df.head(5)
        #user_input_df['job_title'] = user_input_df['job_title'].round(3)
        #st.write(user_input_df)

        # select last row in dataframe
        #red_circle = output_job.loc[6411]
        #red_circle
        #user_input_df

        # Create a graph of recommended job title as a red circle and all other people as lightgray circles
        #job_plot = alt.Chart(output_job, title = alt.Title(str(job),anchor='middle',orient='top')).mark_circle(size=60).encode(
        #x='Years of Experience:Q',
        #y='Salary:Q',
        #color = alt.condition(alt.datum.user_input_df, alt.value('red'), alt.value('blue')),
        #tooltip=['Age', 'Gender', 'Education Level', 'Years of Experience', 'Salary']
        #).interactive()
        #st.altair_chart(job_plot, theme=None)









