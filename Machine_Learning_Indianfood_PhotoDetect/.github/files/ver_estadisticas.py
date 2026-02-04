import json
import os

filepath = os.path.join(os.path.dirname(__file__), 'estadisticas_k2.json')

with open(filepath, 'r') as f:
    data = json.load(f)

print("\n" + "="*60)
print("RESUMEN DE ESTADÍSTICAS POR K")
print("="*60 + "\n")

for K in sorted(data.keys(), key=lambda x: int(x)):
    stats = data[K]
    accuracy = stats['accuracy']
    f1 = stats['f1_weighted']
    precision = stats['precision_weighted']
    recall = stats['recall_weighted']
    
    print(f"K={K:5s} | Accuracy: {accuracy:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

print("\n" + "="*60)
print("Mejor K por Accuracy:")
best_k = max(data.keys(), key=lambda k: data[k]['accuracy'])
best_acc = data[best_k]['accuracy']
print(f"K={best_k}: {best_acc:.4f}")
print("="*60 + "\n")
