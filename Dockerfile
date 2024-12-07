# Use the official Python image as the base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file initially to leverage Docker chacing
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose the port that the Flasj app will run on
EXPOSE 5000

# Set the environment variabel for Flask
ENV FLASK_APP=run.py
ENV FLASK_ENV=production

# Run the application
CMD ["python", "run.py"]
