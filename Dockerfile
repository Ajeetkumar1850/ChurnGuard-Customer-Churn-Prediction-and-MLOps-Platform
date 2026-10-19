FROM python:3.11-slim

WORKDIR /app

# Copy dependency list
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

EXPOSE 5000

CMD ["python", "-m", "flask", "--app", "api.main", "run", "--host", "0.0.0.0", "--port", "5000"]