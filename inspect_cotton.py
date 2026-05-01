import tensorflow as tf

model = tf.keras.models.load_model('models/cotton_model.keras')

print("Top level model layers:")
for layer in model.layers:
    print(f"- {layer.name} ({type(layer).__name__})")
    
inner_model = None
for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        inner_model = layer
        break

if inner_model:
    print("\nInner model layers (last 20):")
    for layer in inner_model.layers[-20:]:
        print(f"- {layer.name} ({type(layer).__name__})")
else:
    print("\nNo inner model found.")
