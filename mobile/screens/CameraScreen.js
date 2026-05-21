import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Image,
  ActivityIndicator,
  Alert,
  Dimensions,
  ScrollView,
  SafeAreaView,
} from 'react-native';

import * as ImagePicker from 'expo-image-picker';
import { detectDisease } from '../services/api';

const { width } = Dimensions.get('window');

export default function CameraScreen({ navigation }) {
  console.log("CAMERASCREEN CHARGE");
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    (async () => {
      await ImagePicker.requestCameraPermissionsAsync();
      await ImagePicker.requestMediaLibraryPermissionsAsync();
    })();
  }, []);

  // Galerie
  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      allowsEditing: true,
      quality: 0.8,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
    }
  };

  // Camera
  const takePhoto = async () => {
    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      quality: 0.8,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
    }
  };

  // ANALYSE
  const analyzeImage = async () => {
    console.log("BOUTON ANALYSER CLIQUE");

    if (!image) {
      Alert.alert("Erreur", "Choisis une image");
      return;
    }

    try {
      setLoading(true);

      console.log("ENVOI IMAGE :", image);

      const result = await detectDisease(image);

      console.log("REPONSE API :", result);

      navigation.navigate('Result', {
        imageUri: image,
        result: result,
      });

    } catch (error) {
      console.log("ERREUR ANALYSE :", error);

      Alert.alert(
        "Erreur",
        "Impossible de contacter le serveur"
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>

        <Text style={styles.title}>
          🌿 Détection des maladies
        </Text>

        <View style={styles.preview}>
          {image ? (
            <Image source={{ uri: image }} style={styles.image} />
          ) : (
            <Text>Aucune image</Text>
          )}
        </View>

        <View style={styles.buttons}>
          <TouchableOpacity style={styles.btn} onPress={takePhoto}>
            <Text style={styles.btnText}>📷 Camera</Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.btn} onPress={pickImage}>
            <Text style={styles.btnText}>🖼️ Galerie</Text>
          </TouchableOpacity>
        </View>

        <TouchableOpacity
          style={styles.analyzeBtn}
          onPress={analyzeImage}
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.analyzeText}>
              Analyser
            </Text>
          )}
        </TouchableOpacity>

      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },

  content: {
    alignItems: 'center',
    padding: 20,
  },

  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
  },

  preview: {
    width: width - 40,
    height: width - 40,
    backgroundColor: '#ddd',
    borderRadius: 10,
    justifyContent: 'center',
    alignItems: 'center',
    overflow: 'hidden',
    marginBottom: 20,
  },

  image: {
    width: '100%',
    height: '100%',
  },

  buttons: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 20,
  },

  btn: {
    backgroundColor: '#3498db',
    padding: 12,
    borderRadius: 10,
  },

  btnText: {
    color: '#fff',
    fontWeight: 'bold',
  },

  analyzeBtn: {
    backgroundColor: '#27ae60',
    padding: 15,
    borderRadius: 10,
    width: '100%',
    alignItems: 'center',
  },

  analyzeText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 16,
  },
});