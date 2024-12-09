#include <Arduino.h>
#include "IntentProcessor.h"
#include "Speaker.h"
#include <HTTPClient.h>
IntentProcessor::IntentProcessor(Speaker *speaker)
{
    m_speaker = speaker;
}
String triggerURL(String url) {
    if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;
        http.begin(url);
        int httpResponseCode = http.GET();

        if (httpResponseCode > 0) {
            String payload = http.getString();
            http.end();
            return payload; // Return the response from the server
        } else {
            Serial.printf("Error on HTTP request: %d\n", httpResponseCode);
            http.end();
            return "HTTP Request Failed";
        }
    } else {
        return "WiFi Disconnected";
    }
}
IntentResult IntentProcessor::turnOnDevice(const Intent &intent)
{
    if (intent.intent_confidence < 0.8)
    {
        Serial.printf("Only %.f%% certain on intent\n", 100 * intent.intent_confidence);
        return FAILED;
    }
    if (intent.device_name.empty())
    {

        Serial.println("No device found");
        return FAILED;
    }
    if (intent.device_name == "office")
    {
        // Trigger the Power Toggle URL
        String response = triggerURL("http://192.168.1.239/cm?cmnd=Power%20Toggle");
        Serial.println("Response: " + response);
        return SUCCESS;
    }
    if (intent.device_name == "dining room")
    {
        // Trigger the Power Toggle URL
        String response = triggerURL("http://192.168.1.113/cm?cmnd=Power%20Toggle");
        Serial.println("Response: " + response);
        return SUCCESS;
    }
    if (intent.device_name == "island")
    {
        // Trigger the Power Toggle URL
        String response = triggerURL("http://192.168.1.68/cm?cmnd=Power%20Toggle");
        Serial.println("Response: " + response);
        return SUCCESS;
    }
    if (intent.device_name == "hall")
    {
        // Trigger the Power Toggle URL
        String response = triggerURL("http://192.168.1.140/cm?cmnd=Power%20Toggle");
        Serial.println("Response: " + response);
        return SUCCESS;
    }
    if (intent.device_confidence < 0.8)
    {
        Serial.printf("Only %.f%% certain on device\n", 100 * intent.device_confidence);
        return FAILED;
    }
    if (intent.trait_value.empty())
    {
        Serial.println("Can't work out the intent action");
        return FAILED;
    }
    if (intent.trait_confidence < 0.8)
    {
        Serial.printf("Only %.f%% certain on trait\n", 100 * intent.trait_confidence);
        return FAILED;
    }
    bool is_turn_on = intent.trait_value == "on";

    // global device name "lights"
    if (intent.device_name == "lights")
    {
        for (const auto &dev_pin : m_device_to_pin)
        {
            digitalWrite(dev_pin.second, is_turn_on);
        }
    }
    else
    {
        // see if the device name is something we know about
        if (m_device_to_pin.find(intent.device_name) == m_device_to_pin.end())
        {
            Serial.printf("Don't recognise the device '%s'\n", intent.device_name.c_str());
            return FAILED;
        }
        digitalWrite(m_device_to_pin[intent.device_name], is_turn_on);
    }
    // success!
    return SUCCESS;
}

IntentResult IntentProcessor::tellJoke()
{
    m_speaker->playRandomJoke();
    Serial.printf("Here is a joke: Hahaha\n");

    return SILENT_SUCCESS;
}

IntentResult IntentProcessor::life()
{
    m_speaker->playLife();
    return SILENT_SUCCESS;
}


IntentResult IntentProcessor::processIntent(const Intent &intent)
{
    if (intent.text.empty())
    {
        Serial.println("No text recognised");
        return FAILED;
    }
    Serial.printf("I heard \"%s\"\n", intent.text.c_str());
    if (intent.intent_name.empty())
    {
        Serial.println("Can't work out what you want to do with the device...");
        return FAILED;
    }
    Serial.printf("Intent is %s\n", intent.intent_name.c_str());
    // if (intent.intent_name == "Turn_on_device")
    if (intent.intent_name == "turn_off_and_on")
    {
        return turnOnDevice(intent);
    }
    if (intent.intent_name == "Tell_joke")
    {
        return tellJoke();
    }
    // if (intent.intent_name == "Life")
    if (intent.intent_name == "Life")
    {
        return life();
    }

    return FAILED;
}

void IntentProcessor::addDevice(const std::string &name, int gpio_pin)
{
    m_device_to_pin.insert(std::make_pair(name, gpio_pin));
    pinMode(gpio_pin, OUTPUT);
}
