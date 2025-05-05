# استخدم بيئة بايثون الرسمية
FROM python:3.10-slim

# تعيين مجلد العمل داخل الحاوية
WORKDIR /app

# نسخ كل الملفات إلى الحاوية
COPY . .

# تثبيت المتطلبات
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# تعيين المنفذ اللي هيشتغل عليه التطبيق
EXPOSE 7860

# أمر التشغيل: استخدم gunicorn لتشغيل app داخل index.py
CMD ["gunicorn", "index:app", "--bind", "0.0.0.0:7860"]
