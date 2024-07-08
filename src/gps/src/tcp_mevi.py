import socket
s = socket.socket()
port = 9000
ip_address_emlid_rover = '192.168.105.180'
s.connect((ip_address_emlid_rover,port))
while True:
	data = s.recv(1024)
	#print(data)
	lat = float(data[26:39])
	lon = float(data[40:54])
	
#	print(lat)
	#print(f"{lat} | {lon}")
	print("latitude: %f | longitude: %f" % (lat, lon))
s.close()
