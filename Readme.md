# Housing Inventory 

## Pre-Requisites
- Clone this repository to your machine
- Install Docker and Docker Compose so you can run the application

## Run the Application
- Start Docker
- Open a terminal and navigate to where you've cloned the folder
- Type `docker  compose -f compose-scs.yml up`
- Run the application
  - [local RStudio version](http://localhost:8787/)
  - [local Jupyter Notebook version](http://localhost:8888/)
- Open another terminal and navigate to where you've cloned the folder
- Type `docker  compose -f compose-scs.yml down`


## What About Tweaking DB directly?
- Open the pre-packaged [SQLite Browser](http://localhost:3000)