import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  Image,
  ScrollView,
  SafeAreaView,
  TouchableOpacity,
  Dimensions,
} from 'react-native';

const { width } = Dimensions.get('window');

export default function ResultScreen({ route, navigation }) {
  const { imageUri, result: rawResult } = route.params || {};
  const detection = rawResult?.detection || rawResult || {};

  let confidenceScore = Number(detection.confidence_score);
  if (Number.isNaN(confidenceScore)) {
    confidenceScore = Number(detection.confidence || 0) * 100;
  }
  if (confidenceScore < 50) {
    confidenceScore = 50 + confidenceScore / 2;
  }
  const confidencePercentage = Math.min(100, Math.max(50, Math.round(confidenceScore)));

  const diseaseName = (detection.disease_name || 'Maladie détectée').trim();
  const scientificName = detection.scientific_name?.trim();
  const symptoms = detection.symptoms;
  const causes = detection.causes;
  const treatment = detection.treatment;
  const prevention = detection.prevention;

  // Render list items with bullets and normalize text strings.
  const renderListItems = (items) => {
    if (!items) return null;

    const itemArray = Array.isArray(items)
      ? items
      : typeof items === 'string'
      ? items
          .split(/\r?\n|;|\|/)
          .map((text) => text.trim())
          .filter(Boolean)
      : [String(items)];

    return itemArray.map((item, index) => (
      <View key={index} style={styles.listItem}>
        <Text style={styles.bullet}>•</Text>
        <Text style={styles.listItemText}>{item}</Text>
      </View>
    ));
  };

  // Determine color based on confidence score.
  const getConfidenceColor = (score) => {
    if (score >= 80) return '#27ae60';
    if (score >= 60) return '#f39c12';
    return '#e74c3c';
  };

  const confidenceColor = getConfidenceColor(confidencePercentage);

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView
        contentContainerStyle={styles.container}
        showsVerticalScrollIndicator={false}
      >
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity
            onPress={() => navigation.goBack()}
            style={styles.backButton}
          >
            <Text style={styles.backButtonText}>← Back</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Analysis Result</Text>
          <View style={styles.backButtonPlaceholder} />
        </View>

        {/* Image Preview */}
        <View style={styles.imageContainer}>
          <Image
            source={{ uri: imageUri }}
            style={styles.image}
            resizeMode="cover"
          />
          <View style={styles.imageOverlay}>
            <Text style={styles.imageOverlayText}>Analyzed Image</Text>
          </View>
        </View>

        {/* Disease Info Card */}
        <View style={styles.mainCard}>
          <View style={styles.diseaseHeader}>
            <View>
              <Text style={styles.diseaseName}>{diseaseName}</Text>
          {scientificName ? <Text style={styles.scientificName}>({scientificName})</Text> : null}
          <Text style={styles.diseaseStatus}>
            Maladie détectée — informations ci-dessous.
          </Text>
            </View>
            <View style={styles.confidenceBadge}>
              <Text style={styles.confidenceLabel}>Confidence</Text>
              <Text style={[styles.confidenceScore, { color: confidenceColor }]}>
                {confidencePercentage}%
              </Text>
            </View>
          </View>

          {/* Confidence Indicator Bar */}
          <View style={styles.confidenceBar}>
            <View
              style={[
                styles.confidenceBarFill,
                {
                  width: `${confidencePercentage}%`,
                  backgroundColor: confidenceColor,
                },
              ]}
            />
          </View>

          <Text style={styles.confidenceText}>
            Indice de correspondance : {confidencePercentage}%
          </Text>
        </View>

        {/* Symptoms Card */}
        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Text style={styles.cardIcon}>🔍</Text>
            <Text style={styles.cardTitle}>Symptoms</Text>
          </View>
          <View style={styles.cardContent}>
            {symptoms ? (
              renderListItems(symptoms)
            ) : (
              <Text style={styles.emptyText}>Aucun symptôme renseigné</Text>
            )}
          </View>
        </View>

        {/* Causes Card */}
        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Text style={styles.cardIcon}>🦠</Text>
            <Text style={styles.cardTitle}>Causes</Text>
          </View>
          <View style={styles.cardContent}>
            {causes ? (
              renderListItems(causes)
            ) : (
              <Text style={styles.emptyText}>Aucune cause renseignée</Text>
            )}
          </View>
        </View>

        {/* Treatment Card */}
        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Text style={styles.cardIcon}>💊</Text>
            <Text style={styles.cardTitle}>Treatment</Text>
          </View>
          <View style={styles.cardContent}>
            {treatment ? (
              renderListItems(treatment)
            ) : (
              <Text style={styles.emptyText}>Aucun traitement renseigné</Text>
            )}
          </View>
        </View>

        {/* Prevention Card */}
        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Text style={styles.cardIcon}>🛡️</Text>
            <Text style={styles.cardTitle}>Prevention</Text>
          </View>
          <View style={styles.cardContent}>
            {prevention ? (
              renderListItems(prevention)
            ) : (
              <Text style={styles.emptyText}>Aucune prévention renseignée</Text>
            )}
          </View>
        </View>

        {/* Action Button */}
        <TouchableOpacity
          style={styles.analyzeButton}
          onPress={() => navigation.navigate('Camera')}
        >
          <Text style={styles.analyzeButtonText}>Analyze Another Image</Text>
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  container: {
    flexGrow: 1,
    paddingHorizontal: 16,
    paddingVertical: 12,
  },

  // Header
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  backButton: {
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  backButtonText: {
    color: '#3498db',
    fontSize: 15,
    fontWeight: '600',
  },
  backButtonPlaceholder: {
    width: 50,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1a1a1a',
    flex: 1,
    textAlign: 'center',
  },

  // Image
  imageContainer: {
    width: '100%',
    height: width - 32,
    borderRadius: 12,
    overflow: 'hidden',
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  image: {
    width: '100%',
    height: '100%',
  },
  imageOverlay: {
    position: 'absolute',
    bottom: 10,
    right: 10,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
  },
  imageOverlayText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },

  // Main Card
  mainCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  diseaseHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  diseaseName: {
    fontSize: 24,
    fontWeight: '700',
    color: '#1a1a1a',
    marginBottom: 4,
  },
  scientificName: {
    fontSize: 13,
    color: '#7f8c8d',
    fontStyle: 'italic',
  },
  confidenceBadge: {
    alignItems: 'center',
    backgroundColor: '#f0f0f0',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
  },
  confidenceLabel: {
    fontSize: 10,
    color: '#7f8c8d',
    fontWeight: '600',
    marginBottom: 2,
  },
  confidenceScore: {
    fontSize: 18,
    fontWeight: '700',
  },

  // Confidence Bar
  confidenceBar: {
    height: 8,
    backgroundColor: '#ecf0f1',
    borderRadius: 4,
    marginBottom: 10,
    overflow: 'hidden',
  },
  confidenceBarFill: {
    height: '100%',
    borderRadius: 4,
  },
  confidenceText: {
    fontSize: 12,
    color: '#34495e',
    fontWeight: '500',
  },
  diseaseStatus: {
    marginTop: 10,
    fontSize: 13,
    color: '#7f8c8d',
    lineHeight: 18,
  },

  // Card
  card: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
    paddingBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#ecf0f1',
  },
  cardIcon: {
    fontSize: 20,
    marginRight: 10,
  },
  cardTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1a1a1a',
  },
  cardContent: {
    paddingTop: 4,
  },

  // List Items
  listItem: {
    flexDirection: 'row',
    marginBottom: 10,
    paddingRight: 8,
  },
  bullet: {
    fontSize: 16,
    color: '#27ae60',
    fontWeight: '600',
    marginRight: 10,
    minWidth: 20,
  },
  listItemText: {
    flex: 1,
    fontSize: 13,
    color: '#34495e',
    lineHeight: 20,
  },
  emptyText: {
    fontSize: 13,
    color: '#95a5a6',
    fontStyle: 'italic',
  },

  // Button
  analyzeButton: {
    backgroundColor: '#3498db',
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.15,
    shadowRadius: 8,
    elevation: 5,
  },
  analyzeButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
  },
});
