from pyinterception import Interception, InterceptionMouse
interception = Interception()
mouse = InterceptionMouse()
interception.set_filter(mouse.device, Interception.FILTER_MOUSE_ALL)
print("Moving mouse right by 50 units…")
mouse.move(50, 0, mouse.device)
