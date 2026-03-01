import React, { useState } from 'react';
import { StyleSheet, Text, View, ScrollView, TouchableOpacity, TextInput, Platform, FlatList } from 'react-native';

const MOCK_DATA = {
    negativeLogits: [
        { text: 'tonsoft', value: -0.68 },
        { text: 'NameInMap', value: -0.65 },
        { text: 'OFDb', value: -0.65 },
        { text: 'thiệu', value: -0.64 },
        { text: 'getItemId', value: -0.61 },
        { text: 'Sympathy', value: -0.59 },
        { text: 'şiv', value: -0.57 },
        { text: 'testy', value: -0.54 },
        { text: 'дописавши', value: -0.54 },
        { text: 'createState', value: -0.53 },
    ],
    positiveLogits: [
        { text: 'success', value: 1.00 },
        { text: 'succeed', value: 0.97 },
        { text: 'réussite', value: 0.97 },
        { text: 'achieving', value: 0.87 },
        { text: 'successes', value: 0.86 },
        { text: 'successful', value: 0.86 },
        { text: 'pursue', value: 0.85 },
        { text: 'achieve', value: 0.85 },
        { text: 'prosper', value: 0.84 },
        { text: 'achievements', value: 0.84 },
    ],
    activations: [
        {
            id: 1,
            label: 'Australians',
            score: 69.16,
            content: [
                { text: "what it's like " },
                { text: "to have giant dreams and intend to help other", highlight: true, intensity: 0.6 },
                { text: " young " },
                { text: "Australians achieve theirs by sharing my", highlight: true, intensity: 0.9 },
                { text: " knowledge and developing " },
                { text: "their", highlight: true, intensity: 0.5 },
                { text: " talent " },
                { text: "in", highlight: true, intensity: 0.3 },
                { text: " collaboration with The X" }
            ]
        },
        {
            id: 2,
            label: 'should',
            score: 68.57,
            content: [
                { text: "support and guide him " },
                { text: "towards success", highlight: true, intensity: 0.8 },
                { text: ". The movie is simply a " },
                { text: "reminder that you should never let go of your dream", highlight: true, intensity: 0.9 },
                { text: " and" }
            ]
        },
        {
            id: 3,
            label: 'to',
            score: 67.19,
            content: [
                { text: ". God " },
                { text: "has", highlight: true, intensity: 0.4 },
                { text: " been good " },
                { text: "to", highlight: true, intensity: 0.5 },
                { text: " that " },
                { text: "little boy", highlight: true, intensity: 0.3 },
                { text: " from long " },
                { text: "ago, enabling me to do something", highlight: true, intensity: 0.7 },
                { text: " I love" },
                { text: ". I am", highlight: true, intensity: 0.5 },
                { text: " amazed and thankful every day." }
            ]
        }
    ]
};

const API_BASE = 'http://localhost:8000';

export default function FeatureDetails({ feature, onClose, modelId }) {
  const [testText, setTestText] = useState('');
  const [llmLabel, setLlmLabel] = useState(null);
  const [llmLoading, setLlmLoading] = useState(false);
  const [llmError, setLlmError] = useState(null);

  const fetchLlmLabel = async () => {
    if (!feature?.id || !modelId) return;
    setLlmLoading(true);
    setLlmLabel(null);
    setLlmError(null);
    try {
      const res = await fetch(`${API_BASE}/label-feature`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: modelId,
          feature_idx: feature.id,
          analyzer: 'sae',
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setLlmLabel(data);
    } catch (e) {
      setLlmError(e.message);
    } finally {
      setLlmLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      {/* Header Bar */}
      <View style={styles.topBar}>
        <TouchableOpacity style={styles.doneButton} onPress={onClose}>
            <Text style={styles.doneButtonText}>Done</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.popupButton}>
            <Text style={styles.popupButtonText}>↗ Popup</Text>
        </TouchableOpacity>
      </View>

      <ScrollView style={styles.contentScroll}>
        <View style={styles.mainContent}>
            
            <View style={styles.headerRow}>
                <Text style={styles.pageTitle}>{feature?.description || "phrases that emphasize personal aspirations and empowerment"}</Text>
                <View style={styles.modelCard}>
                    <Text style={styles.modelInfo}>GPT-2 SAE</Text>
                    <Text style={styles.modelDetail}>SPARSE AUTOENCODER</Text>
                    <Text style={styles.modelIndex}>INDEX {feature?.id || '—'}</Text>
                </View>
            </View>

            {/* Logits Section */}
            <View style={styles.logitsContainer}>
                
                {/* Negative Logits */}
                <View style={styles.logitsColumn}>
                    <View style={styles.logitsHeaderRow}>
                         <Text style={styles.logitsTitle}>NEGATIVE LOGITS ⓘ</Text>
                    </View>
                    {MOCK_DATA.negativeLogits.map((item, idx) => (
                        <View key={idx} style={styles.logitRow}>
                            <View style={[styles.logitPill, styles.negativePill]}>
                                <Text style={styles.logitText}>{item.text}</Text>
                            </View>
                            <Text style={styles.logitValue}>{item.value.toFixed(2)}</Text>
                        </View>
                    ))}
                </View>

                {/* Positive Logits */}
                <View style={styles.logitsColumn}>
                     <View style={styles.logitsHeaderRow}>
                         <Text style={styles.logitsTitle}>POSITIVE LOGITS ⓘ</Text>
                    </View>
                    {MOCK_DATA.positiveLogits.map((item, idx) => (
                        <View key={idx} style={styles.logitRow}>
                            <View style={[styles.logitPill, styles.positivePill]}>
                                <Text style={styles.logitText}>{item.text}</Text>
                            </View>
                            <Text style={styles.logitValue}>{item.value.toFixed(2)}</Text>
                        </View>
                    ))}
                </View>

                {/* Histogram (Simplified) */}
                <View style={styles.histogramColumn}>
                    <Text style={styles.histogramTitle}>ACTIVATIONS DENSITY 0.290% ⓘ</Text>
                    <View style={styles.histogramChart}>
                         {/* Mock bars */}
                         {[...Array(20)].map((_, i) => (
                             <View key={i} style={[
                                 styles.bar, 
                                 { 
                                     height: Math.random() * 80 + 10,
                                     backgroundColor: i < 10 ? '#ffb3b3' : '#a3bffa' // Reddish then Bluish
                                 }
                             ]} />
                         ))}
                    </View>
                    <View style={styles.xAxis}>
                        <Text style={styles.axisLabel}>-0.5</Text>
                        <Text style={styles.axisLabel}>0</Text>
                        <Text style={styles.axisLabel}>0.5</Text>
                    </View>
                </View>

            </View>

            {/* LLM Label */}
            <View style={styles.llmSection}>
              <TouchableOpacity
                style={[styles.actionButton, styles.llmButton]}
                onPress={fetchLlmLabel}
                disabled={llmLoading}
              >
                <Text style={styles.actionButtonText}>
                  {llmLoading ? '…Labeling' : '✦ Get LLM Label'}
                </Text>
              </TouchableOpacity>
              {llmError && (
                <Text style={styles.llmError}>{llmError}</Text>
              )}
              {llmLabel && (
                <View style={styles.llmResult}>
                  <View style={styles.llmLabelRow}>
                    <Text style={styles.llmLabelText}>{llmLabel.label}</Text>
                    <View style={[styles.confidenceBadge,
                      llmLabel.confidence === 'high' ? styles.confHigh
                      : llmLabel.confidence === 'medium' ? styles.confMed
                      : styles.confLow
                    ]}>
                      <Text style={styles.confidenceText}>{llmLabel.confidence?.toUpperCase()}</Text>
                    </View>
                  </View>
                  <Text style={styles.llmExplanation}>{llmLabel.explanation}</Text>
                  {llmLabel.top_tokens?.length > 0 && (
                    <View style={styles.tokenChips}>
                      {llmLabel.top_tokens.slice(0, 10).map((tok, i) => (
                        <View key={i} style={styles.chip}>
                          <Text style={styles.chipText}>{tok}</Text>
                        </View>
                      ))}
                    </View>
                  )}
                </View>
              )}
            </View>

            {/* Test Activation */}
            <View style={styles.testSection}>
                <TextInput 
                    style={styles.testInput}
                    placeholder="Test activation with custom text."
                    value={testText}
                    onChangeText={setTestText}
                />
                <TouchableOpacity style={[styles.actionButton, styles.testButton]}>
                    <Text style={styles.actionButtonText}>▷ Test</Text>
                </TouchableOpacity>
                <TouchableOpacity style={[styles.actionButton, styles.steerButton]}>
                    <Text style={[styles.actionButtonText, styles.steerButtonText]}>⎇ Steer</Text>
                </TouchableOpacity>
            </View>

            {/* Top Activations */}
            <View style={styles.activationsHeader}>
                <Text style={styles.topLabel}>TOP</Text>
                <View style={styles.activationsTag}>
                    <Text style={styles.activationsTagText}>ACTIVATIONS</Text>
                </View>
            </View>

            <View style={styles.activationsList}>
                {MOCK_DATA.activations.map((act) => (
                    <View key={act.id} style={styles.activationCard}>
                        <View style={styles.activationScoreBox}>
                            <Text style={styles.activationScoreLabel}>{act.label}</Text>
                            <Text style={styles.activationScoreValue}>{act.score}</Text>
                        </View>
                        <View style={styles.activationContent}>
                            <Text style={styles.activationText}>
                                {act.content.map((segment, sIdx) => (
                                    <Text key={sIdx} style={{
                                        backgroundColor: segment.highlight ? `rgba(90, 230, 150, ${segment.intensity || 0.5})` : 'transparent'
                                    }}>
                                        {segment.text}
                                    </Text>
                                ))}
                            </Text>
                            <TouchableOpacity style={styles.copyButton}>
                                <Text style={styles.copyIcon}>❑</Text>
                            </TouchableOpacity>
                        </View>
                    </View>
                ))}
            </View>

        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  topBar: {
    padding: 10,
    flexDirection: 'row',
    justifyContent: 'space-between',
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
    backgroundColor: '#f8f9fa',
  },
  doneButton: {
    padding: 8,
    backgroundColor: '#e2e8f0',
    borderRadius: 6,
  },
  doneButtonText: {
    fontWeight: '600',
    color: '#475569',
  },
  popupButton: {
    padding: 8,
    backgroundColor: '#e2e8f0',
    borderRadius: 6,
  },
  popupButtonText: {
    fontWeight: '600',
    color: '#475569',
  },
  contentScroll: {
    flex: 1,
  },
  mainContent: {
    padding: 20,
    maxWidth: 1200,
    alignSelf: 'center',
    width: '100%',
  },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20,
    flexWrap: 'wrap',
    gap: 10,
  },
  pageTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#334155',
    flex: 1,
    minWidth: 300,
  },
  modelCard: {
    backgroundColor: '#eff6ff', // Light blue bg
    padding: 10,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#bfdbfe',
    alignItems: 'flex-end',
  },
  modelInfo: { fontSize: 10, color: '#64748b' },
  modelDetail: { fontSize: 10, color: '#64748b' },
  modelIndex: { fontSize: 12, fontWeight: 'bold', color: '#334155' },

  // LLM label area
  llmSection: {
    marginBottom: 20,
    gap: 10,
  },
  llmButton: {
    backgroundColor: '#7c3aed',
    alignSelf: 'flex-start',
  },
  llmError: {
    color: '#dc2626',
    fontSize: 13,
  },
  llmResult: {
    backgroundColor: '#f5f3ff',
    borderRadius: 8,
    padding: 14,
    borderWidth: 1,
    borderColor: '#ddd6fe',
    gap: 6,
  },
  llmLabelRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    flexWrap: 'wrap',
  },
  llmLabelText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#3730a3',
    flex: 1,
  },
  confidenceBadge: {
    borderRadius: 20,
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  confHigh:  { backgroundColor: '#bbf7d0' },
  confMed:   { backgroundColor: '#fef08a' },
  confLow:   { backgroundColor: '#fecaca' },
  confidenceText: {
    fontSize: 10,
    fontWeight: 'bold',
    color: '#334155',
  },
  llmExplanation: {
    fontSize: 13,
    color: '#475569',
    lineHeight: 20,
  },
  tokenChips: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
    marginTop: 4,
  },
  chip: {
    backgroundColor: '#ede9fe',
    borderRadius: 12,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderWidth: 1,
    borderColor: '#c4b5fd',
  },
  chipText: {
    fontSize: 12,
    color: '#5b21b6',
  },

  logitsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: 20,
    gap: 20,
  },
  logitsColumn: {
    flex: 1,
    minWidth: 200,
  },
  logitsHeaderRow: {
    marginBottom: 10,
  },
  logitsTitle: {
      fontSize: 10,
      fontWeight: 'bold',
      color: '#94a3b8',
      textTransform: 'uppercase',
  },
  logitRow: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'space-between',
      marginBottom: 4,
  },
  logitPill: {
      paddingHorizontal: 6,
      paddingVertical: 2,
      borderRadius: 4,
      marginRight: 8,
  },
  negativePill: { backgroundColor: '#fee2e2' },
  positivePill: { backgroundColor: '#dbeafe' },
  logitText: { fontFamily: 'monospace', fontSize: 12, color: '#0f172a' },
  logitValue: { fontFamily: 'monospace', fontSize: 12, color: '#64748b' },
  
  histogramColumn: {
      flex: 1.5,
      minWidth: 300,
      height: 200,
      justifyContent: 'flex-end',
  },
  histogramTitle: {
      fontSize: 10,
      fontWeight: 'bold',
      color: '#94a3b8',
      textAlign: 'right',
      marginBottom: 10,
  },
  histogramChart: {
      flexDirection: 'row',
      alignItems: 'flex-end',
      height: 150,
      borderBottomWidth: 1,
      borderBottomColor: '#e2e8f0',
      gap: 2,
  },
  bar: {
      flex: 1,
      borderTopLeftRadius: 2,
      borderTopRightRadius: 2,
  },
  xAxis: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      marginTop: 5,
  },
  axisLabel: { fontSize: 10, color: '#94a3b8' },

  testSection: {
      flexDirection: 'row',
      marginBottom: 30,
      gap: 10,
      alignItems: 'center',
  },
  testInput: {
      flex: 1,
      borderWidth: 1,
      borderColor: '#e2e8f0',
      borderRadius: 4,
      padding: 10,
      height: 40,
  },
  actionButton: {
      paddingHorizontal: 16,
      height: 40,
      justifyContent: 'center',
      borderRadius: 4,
  },
  testButton: { backgroundColor: '#0f5385' },
  steerButton: { backgroundColor: '#fff', borderWidth: 1, borderColor: '#22c55e' },
  actionButtonText: { color: '#fff', fontWeight: 'bold' },
  steerButtonText: { color: '#22c55e' },

  activationsHeader: {
      flexDirection: 'row',
      alignItems: 'center',
      marginBottom: 15,
  },
  topLabel: { fontSize: 10, color: '#64748b', marginRight: 5, fontWeight: 'bold' },
  activationsTag: { backgroundColor: '#4ade80', paddingHorizontal: 6, paddingVertical: 2, borderRadius: 2 },
  activationsTagText: { color: '#fff', fontSize: 10, fontWeight: 'bold' },
  
  activationsList: { gap: 10 },
  activationCard: {
      flexDirection: 'row',
      marginBottom: 10,
      alignItems: 'flex-start',
  },
  activationScoreBox: {
      width: 80,
      padding: 5,
      backgroundColor: '#f8fafc',
      marginRight: 10,
      borderRadius: 4,
  },
  activationScoreLabel: { fontSize: 10, fontWeight: 'bold', color: '#475569', marginBottom: 2 },
  activationScoreValue: { fontSize: 12, color: '#166534', fontWeight: 'bold' },
  
  activationContent: {
      flex: 1,
      flexDirection: 'row',
      alignItems: 'flex-start',
  },
  activationText: {
      flex: 1,
      fontSize: 14,
      lineHeight: 22,
      color: '#334155',
      fontFamily: 'monospace'
  },
  copyButton: {
      marginLeft: 10,
      padding: 4,
  },
  copyIcon: {
      fontSize: 16,
      color: '#cbd5e1'
  }

});
