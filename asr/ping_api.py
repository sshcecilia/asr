from flask import Flask

# Create new Flask web application.Argument indicates where to look for 
# resources such as templates and static files.
app = Flask(__name__)

# Route decorator indicates what url to trigger for application
# Use get method since we are only retrieving data
@app.route("/ping", methods = ['GET'])

def ping():
    '''
    ping function returns pong
    '''
    return 'pong'

# Launch the Flask application. Run on port 8001 (default port 5000)
app.run(port = 8001)