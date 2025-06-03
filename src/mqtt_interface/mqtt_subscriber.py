import paho.mqtt.client as mqtt

broker_ip = "192.168.1.61"  # Your Windows PC IP where Mosquitto broker runs

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Connected to MQTT broker.")
        client.subscribe("teep/flood_detection")  # ✅ Correct topic
    else:
        print(f"❌ Failed to connect. Return code: {rc}")

def on_message(client, userdata, msg):
    print(f"📨 {msg.topic}: {msg.payload.decode()}")

def on_disconnect(client, userdata, rc):
    print("🔌 Disconnected from MQTT broker.")

client = mqtt.Client(client_id="PC-Subscriber")
client.on_connect = on_connect
client.on_message = on_message
client.on_disconnect = on_disconnect

client.connect(broker_ip, 1883, 60)
client.loop_forever()
