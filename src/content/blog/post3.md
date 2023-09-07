---
title: "Messing around with arXiv database!"
description: "Exploring postgres vector databases :)"
pubDate: "Sep 06 2023"
heroImage: "/postgres.webp"
---
I got to be honest here. That last blog? Whew, that was very time-consuming. So, I'm switching things up a bit with this post. We're going full-on stream of consciousness mode here. 🧠 As I go along, I want a tablulate the errors I make and how I fixed them. This is how I've been approaching development for sometime now!<br>
I've got this super cool idea for a pet project, and I'm gonna lay it all out as I code away. Fingers crossed it won't eat up as much of my time as that last one did! 😅🕒
<br>
__arXiv__ is a treasure trove of incredible papers, and the best part? It's open to everyone! No hidden papers, no paywalls, it's all out there for anyone to read. Recently, I stumbled upon this Kaggle dataset that contains all the metadata for every paper on arXiv. Now, what I'm cooking up here is a little project to process that Kaggle dataset, whip up some embedding vectors, and then use the Postgres Vector DB to index these vectors.
Imagine someone wanting to do a literature survey or check how novel their idea is. This project could make those tasks a walk in the park. So, buckle up, we're diving in! 🚀

### Let's process the data : 
The first task would be to process the dataset. It is available over [here](https://www.kaggle.com/datasets/Cornell-University/arxiv). I downloaded the data in .json format! Time to process the json with the good old Pandas!
        
        def readJson():
            json_data = pd.read_json('arxiv.json')
            print(json_data.head)

Oof! Some __trailing data issue__ while reading the json file.
Okay so the dataset has '\n' in the abstract, title sections. When pandas tries to read this json format, it encounters end of line and maybe a trailing issue. Better to fix it with __lines = True__ parameter in pd.read_json() function. Now this might take a long time - even the metadata is bulky!<br>
<br>__Bug #1:__ <br>
| <span style = "color:red">Error</span> | <span style = "color:green">Fix</span> |
| :--- | :----------- |
| ValueError: Trailing data | json_data = pd.read_json('arxiv.json',lines=True) |

### Embeddings :
The next task to find a proper embeddings for the dataset. Now for the utility of the project, it is best if we embed the abstract of each json entry. For this task, we use the __Sentence Transformer__ library.

        def get_text_embedding(proposed_idea):
            
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            embedding_idea = model.encode(proposed_idea,convert_to_numpy=True)
            
            return embedding_idea

Great we have the embeddings! If we output the size of the vector, you'll notice that it is of length 384.
(Ooh ooh I think with more complex embeddings like from Llama or Open-AI, the relevancy should increase! I definitely have to do it later)
In the background, I have added a code to read the json file : retrieve all the necessary information, take the abstract, call the get_text_embedding function and finally save all these information in an numpy array! It is a huge dataset! Better to have a backup. Can't be computing vectors each and every time.
### PostgreSQL:
Let us setup PostgreSQL db on __Amazon RDS__. One can set it up locally as well. I wanted to try out the Amazon RDS. So going ahead with it. 
After a lot of clicks, the RDS database is up and running. Made a mistake of not setting up the inbound request in EC2 properly. Was not able to access the DB from my system for quiet a while xD.<br>
<br>__Bug #2:__ <br>
| <span style = "color:red">Error</span> | <span style = "color:green">Fix</span> |
| :--- | :----------- |
| Inbound rule for security group not set on EC2 instance | Follow this [blog](https://saturncloud.io/blog/how-to-add-a-https-inbound-rule-to-a-security-group-on-an-amazon-aws-ec2-instance/) to set appropriate inbound rules|

Installed __DataGrip__ locally to connect to the PostgreSQL server.
Yay! Now to the querys.

        def createDatabase(conn):
            cursor = conn.cursor()
            database_create_query = """ create database paperDb"""
            cursor.execute(database_create_query)
            cursor.close()

        def createTable(conn):
            cursor = conn.cursor()
            table_create_query = """ create table arxivVectorDB(paperId text PRIMARY KEY,title text, authors text, abstract text,embedding vector(384)) """
            cursor.execute(table_create_query)
            cursor.close()
Another error. Encountered a "vector" does not exist error.

The vector is an extension in the Postgres environment. It needs to be added as extension. With rds, I don't have to download but if one is trying this locally, I think we need to do a git based installation and use the create extension query.
<br>__Bug #3:__<br>
| <span style = "color:red">Error</span> | <span style = "color:green">Fix</span> |
| :--- | :----------- |
| type "vector" does not exist |  cursor.execute("CREATE EXTENSION IF NOT EXISTS vector") |
         
Note that these commands can be run from DataGrip as well. Very convenient. Moving on.
Now that we have processed the data from json file, we can create a list of tuples and these can be inserted as a batch into the db using the following code : 

        from psycopg2.extras import execute_values

        def insertIntoTable(conn,list_to_insert):
            cursor = conn.cursor()
            table_insert_batch = execute_values(cursor, "INSERT INTO arxivVectorDB(paperId, title, authors, abstract, embedding) VALUES %s", list_to_insert)
            print("Executed the batch command!! \n")
            cursor.close()
    
<br>__Bug #4:__<br>
| <span style = "color:red">Error</span> | <span style = "color:green">Fix</span> |
| :--- | :----------- |
|pyscopg-2 programming error - can't adapt type numpy.| Issue on the pyscopg2 interface. Need to add adapters. Check [this](https://stackoverflow.com/questions/39564755/programmingerror-psycopg2-programmingerror-cant-adapt-type-numpy-ndarray) out. |

We can check if the data has been put into the db using DataGrip: 
![Data grip interface](/DataGripTable.png)

Cool, now we can get input from the user - the abstract of their idea. We need to build embedding vectors out of the abstract, query the vector db for the abstracts that are similar to the input abstract. To begin with, we shall start with the exact match query using cosine similarity : 

        def searchVectorDatabase(conn,proposed_idea):
            embedding_idea = np.array(get_text_embedding(proposed_idea))
            # Register pgvector extension
            register_vector(conn)
            cursor = conn.cursor()
            cursor.execute("SELECT abstract FROM paperTable ORDER BY embedding <=> %s LIMIT 3", (embedding_idea,))
            result = cursor.fetchall()
            return result       
The above function will return the top 3 abstracts that are similar to the query vector.
I'm curious. So this should probably check the similarity with every vector in the table. Currently we have stopped the database size at 10000. PostgreSQL also offers another solution to be quicker for querying. But as always there is a tradeoff. Here we trade speed for the perfect recall! We shall be using the IVFFlat index. This one basically divides the vectors into sub-indices and then searches these indexes approximately for a match with the query vector.

The github repository outlines the use of such indexes [here](https://github.com/pgvector/pgvector#indexing). We need to check the parameters and use them wisely.

Quoting from the repository : 

    Keys to achieving good recall are:

        1. Choose an appropriate number of lists - a good place to start is rows / 1000 for up to 1M rows and sqrt(rows) for over 1M rows
        2. When querying, specify an appropriate number of probes (higher is better for recall, lower is better for speed) - a good place to start is sqrt(lists)

Thus we'll be indexing with 10000/1000 = 10 for the list parameter and for the probe parameter we'll stick with 100.

Voila!

![Output](/FinalOutput.png)


Alright, setting aside the acceleration in time, the style of writing pretty much documented how I approached coding this small project. Do reach out to me if you have any doubts or any other exciting idea to build on top of this!

Click [here](https://github.com/bhargav191098/arXiV_VectorDB) to check out the github repository.

Cheers! :)