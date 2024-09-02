from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.properties import StringProperty
from kivy.utils import platform
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDRaisedButton
from kivymd.app import MDApp
from plyer import accelerometer, gps
from joblib import load
import numpy as np
import os
from jnius import autoclass, PythonJavaClass, java_method, cast


model_path = os.path.join(os.getcwd(), 'models', 'anomaly_detection_model.pkl')
model = load(model_path)


Context = autoclass('android.content.Context')
LocationManager = autoclass('android.location.LocationManager')
SmsManager = autoclass('android.telephony.SmsManager')
PythonActivity = autoclass('org.kivy.android.PythonActivity')
ActivityCompat = autoclass('androidx.core.app.ActivityCompat')
PackageManager = autoclass('android.content.pm.PackageManager')



PERMISSIONS = [
    "android.permission.SEND_SMS",
    "android.permission.ACCESS_FINE_LOCATION",
    "android.permission.RECEIVE_BOOT_COMPLETED",
]



class GpsHelper(PythonJavaClass):
    __javainterfaces__ = ['android/location/LocationListener']

    def get_location(self):
        location_manager = PythonActivity.mActivity.getSystemService(Context.LOCATION_SERVICE)
        location = location_manager.getLastKnownLocation(LocationManager.GPS_PROVIDER)
        if location:
            return location.getLatitude(), location.getLongitude()
        else:
            return None, None

    def send_sms(self, phone_number, message):
        sms_manager = SmsManager.getDefault()
        sms_manager.sendTextMessage(phone_number, None, message, None, None)

class SOSApp(MDApp):
    notification_text = StringProperty("")

    def build(self):
        self.gps_helper = GpsHelper()
        self.dialog = None
        return BoxLayout()

    def on_start(self):
        self.check_permissions()
        accelerometer.enable()
        Clock.schedule_interval(self.detect_anomaly, 1)  

    def detect_anomaly(self, dt):
        if accelerometer.is_enabled():
            accel = accelerometer.acceleration
            if accel:
                accel_array = np.array([accel[0], accel[1], accel[2]]).reshape(1, -1)
                prediction = model.predict(accel_array)
                if prediction == 1:  
                    self.trigger_sos()

    def trigger_sos(self):
        self.show_notification()
        Clock.schedule_once(self.check_user_response, 30)  

    def show_notification(self):
        if not self.dialog:
            self.dialog = MDDialog(
                title="SOS Alert",
                text="Abnormal movement detected! Click to cancel.",
                buttons=[
                    MDRaisedButton(text="OK", on_release=self.cancel_sos)
                ]
            )
        self.dialog.open()

    def cancel_sos(self, *args):
        if self.dialog:
            self.dialog.dismiss()
            self.dialog = None

    def check_user_response(self, dt):
        if self.dialog:
            self.dialog.dismiss()
            self.dialog = None
            self.send_sos_sms()

    def send_sos_sms(self):
        latitude, longitude = self.gps_helper.get_location()
        if latitude and longitude:
            message = f"SOS Alert: Abnormal movement detected. Location: https://maps.google.com/?q={latitude},{longitude}"
            contacts = ["+919042089926", "+917418760410", "+918589027615"]  
            for contact in contacts:
                self.gps_helper.send_sms(contact, message)

    def check_permissions(self):
        for permission in PERMISSIONS:
            if ActivityCompat.checkSelfPermission(PythonActivity.mActivity, permission) != PackageManager.PERMISSION_GRANTED:
                ActivityCompat.requestPermissions(PythonActivity.mActivity, [permission], 1)


if __name__ == '__main__':
    SOSApp().run()
