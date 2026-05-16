from app import create_app

# Create the application instance using the factory
app = create_app()

if __name__ == "__main__":
    # In development, we run with debug mode. 
    # In production, this file is called by a WSGI server like Gunicorn.
    app.run(host="0.0.0.0", port=5000, debug=True)