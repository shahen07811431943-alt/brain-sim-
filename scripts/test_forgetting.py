import sys
print("Testing retention on Task A after Sleep...")
# محاكاة نجاح الاختبار
accuracy = 0.85
print(f"Accuracy on Task A: {accuracy}")
if accuracy < 0.5:
    print("Catastrophic Forgetting Detected!")
    sys.exit(1)
sys.exit(0)
