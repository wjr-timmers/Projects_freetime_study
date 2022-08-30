## Project Rooster

This is a project I worked on during my work as Coordinator of the IT Studentenpool at the VU. Scheduling a team of 30+ students across different departments was one of my duties. Because this process took a lot of time I've made an attempt to automate this process using python. 

The implementation consists of a few scripts:
- beschikbaarheid.py gets the availability of students from an outlook import and transforms it to a workable pandas dataframe
- EA.py is the evolutionary algorithm I implemented to look for an optimal schedule solution
- main.py runs the main script, from getting the data to producing a schedule
- schedule.py contains helper function to generate for example an empty / random schedule
- skillmatrix.py reads a file that indicates the capabilities and skills of every student