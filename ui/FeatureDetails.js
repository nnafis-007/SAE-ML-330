import React, { useState } from 'react';
import {
  ActivityIndicator,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';

const API_BASE = 'http://localhost:8000';
const PEOPLE_SPEECH_DATASET = 'MLCommons/peoples_speech';
const PEOPLE_SPEECH_CONFIG = 'validation';
const PEOPLE_SPEECH_SPLIT = 'validation';

function sanitizeDecimalInput(value) {
  const cleaned = (value || '').replace(/[^0-9.]/g, '');
  const firstDotIndex = cleaned.indexOf('.');
  if (firstDotIndex === -1) return cleaned;
  return cleaned.slice(0, firstDotIndex + 1) + cleaned.slice(firstDotIndex + 1).replace(/\./g, '');
}

export default function FeatureDetails({ feature, onClose, modelId }) {
  const [maxSentences, setMaxSentences] = useState('');
  const [activatingExamples, setActivatingExamples] = useState('');
  const [minActivation, setMinActivation] = useState('0.1');
  const [pageSize] = useState(25);
  const [matchesPage, setMatchesPage] = useState(1);
  const [lastActivationQuery, setLastActivationQuery] = useState(null);

  const [llmLabel, setLlmLabel] = useState(null);
  const [llmLoading, setLlmLoading] = useState(false);
  const [pageLoading, setPageLoading] = useState(false);
  const [llmError, setLlmError] = useState(null);

  const [activationPayload, setActivationPayload] = useState(null);
  const [analysisMeta, setAnalysisMeta] = useState(null);

  const promptAlignedRows = (llmLabel?.llm_prompt_examples || []).map((ex, idx) => ({
    id: `prompt-${idx}`,
    context: ex.context,
    activation: ex.activation,
  }));
  const usePromptAlignedRows = llmLabel?.llm_prompt_examples?.length > 0;

  const loadActivationPage = async (query, pageNo) => {
    setPageLoading(true);
    try {
      const res = await fetch(`${API_BASE}/feature-activations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...query, page: pageNo, page_size: pageSize }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setActivationPayload(data);
      setMatchesPage(data.page || pageNo);
      return data;
    } finally {
      setPageLoading(false);
    }
  };

  const fetchLlmLabel = async () => {
    if (!feature?.id || !modelId) {
      return;
    }

    const maxSentencesTrimmed = maxSentences.trim();
    const activatingExamplesTrimmed = activatingExamples.trim();

    const sentenceLimit =
      maxSentencesTrimmed === '' ? null : Math.max(1, Number(maxSentencesTrimmed) || 0);
    const activatingExamplesLimit =
      activatingExamplesTrimmed === '' ? null : Math.max(1, Number(activatingExamplesTrimmed) || 0);

    if (sentenceLimit !== null && !Number.isFinite(sentenceLimit)) {
      setLlmError('Number of sentences must be a valid positive integer.');
      return;
    }
    if (activatingExamplesLimit !== null && !Number.isFinite(activatingExamplesLimit)) {
      setLlmError('Number of activating examples must be a valid positive integer.');
      return;
    }

    // Fallback if user leaves both blank.
    const effectiveSentenceLimit = sentenceLimit ?? 200;
    const minAct = Math.max(0, Number(minActivation) || 0);
    setLlmLoading(true);
    setLlmLabel(null);
    setLlmError(null);
    setActivationPayload(null);
    setAnalysisMeta(null);

    try {
      // Fetch activating contexts from HF People's Speech.
      // Rules implemented:
      // 1) both provided => enforce both;
      // 2) activating examples only => auto-expand scanned sentences until enough activating sentences;
      // 3) sentences only => use those scanned sentences, then send activating-only sentence subset to LLM.
      const requiredActivatingSentences = activatingExamplesLimit;
      const hasExplicitSentenceLimit = sentenceLimit !== null;
      const scanSentences = hasExplicitSentenceLimit ? effectiveSentenceLimit : null;

      const activationQuery = {
        model_id: modelId,
        feature_id: feature.id,
        dataset_name: PEOPLE_SPEECH_DATASET,
        dataset_config: PEOPLE_SPEECH_CONFIG,
        split: PEOPLE_SPEECH_SPLIT,
        max_sentences: scanSentences,
        target_activating_examples: !hasExplicitSentenceLimit ? requiredActivatingSentences : null,
        min_activation: minAct,
        page: 1,
        page_size: pageSize,
      };

      let activationData = await loadActivationPage(activationQuery, 1);
      setLastActivationQuery(activationQuery);

      let uniqueSentences = activationData.activating_sentences || [];

      if (uniqueSentences.length === 0) {
        throw new Error('No activating contexts found with the selected thresholds.');
      }

      if (!hasExplicitSentenceLimit && requiredActivatingSentences && uniqueSentences.length < requiredActivatingSentences) {
        throw new Error(
          `Could not find ${requiredActivatingSentences} activating sentence(s). Try lowering minimum activation.`
        );
      }

      const corpusForLlm = requiredActivatingSentences
        ? uniqueSentences.slice(0, requiredActivatingSentences)
        : uniqueSentences;

      setActivationPayload(activationData);

      // Step 2: run LLM labeling using only those sampled sentences.
      const labelRes = await fetch(`${API_BASE}/label-feature`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: modelId,
          feature_idx: feature.id,
          analyzer: 'sae',
          corpus_texts: corpusForLlm,
          labeling_config: {
            top_k: requiredActivatingSentences ?? 0,
            skip_first_token: true,
            min_activation: minAct,
            num_sentences: scanSentences,
          },
        }),
      });

      if (!labelRes.ok) {
        const err = await labelRes.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${labelRes.status}`);
      }

      const labelData = await labelRes.json();
      setLlmLabel(labelData);
      setAnalysisMeta({
        requestId: labelData.request_id || null,
        llmActivationMode: labelData.llm_activation_mode || 'standardize',
        sentenceLimit: hasExplicitSentenceLimit ? effectiveSentenceLimit : null,
        scannedSentences: activationData.scanned_sentences,
        activatingExamplesLimit: requiredActivatingSentences,
        minAct,
        corpusSentencesUsed: corpusForLlm.length,
        totalMatches: activationData.total_matches || 0,
        totalPages: activationData.total_pages || 1,
        dataset: `${PEOPLE_SPEECH_DATASET} [${PEOPLE_SPEECH_SPLIT}]`,
      });
    } catch (e) {
      setLlmError(e.message);
    } finally {
      setLlmLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.topBar}>
        <TouchableOpacity style={styles.doneButton} onPress={onClose}>
          <Text style={styles.doneButtonText}>Done</Text>
        </TouchableOpacity>
      </View>

      <ScrollView style={styles.contentScroll}>
        <View style={styles.mainContent}>
          <View style={styles.headerRow}>
            <Text style={styles.pageTitle}>{feature?.description || `Feature ${feature?.id ?? ''}`}</Text>
            <View style={styles.modelCard}>
              <Text style={styles.modelInfo}>GPT-2 SAE</Text>
              <Text style={styles.modelDetail}>LLM ANALYSIS</Text>
              <Text style={styles.modelIndex}>FEATURE {feature?.id || '-'}</Text>
            </View>
          </View>

          <View style={styles.analysisPanel}>
            <Text style={styles.panelTitle}>Run Feature LLM Analysis</Text>
            <Text style={styles.panelHint}>
              Dataset is fixed to Hugging Face People\'s Speech. Provide sentence count and/or activating-example count.
              If only activating examples is provided, scanning auto-expands until enough activating sentences are found.
            </Text>

            <View style={styles.inputRow}>
              <View style={styles.inputField}>
                <Text style={styles.inputLabel}>How many sentences to analyze</Text>
                <TextInput
                  style={styles.numericInput}
                  value={maxSentences}
                  onChangeText={(t) => setMaxSentences(t.replace(/[^0-9]/g, ''))}
                  keyboardType="numeric"
                  placeholder="Optional (e.g. 200)"
                />
              </View>
              <View style={styles.inputField}>
                <Text style={styles.inputLabel}>Activating examples to send to LLM</Text>
                <TextInput
                  style={styles.numericInput}
                  value={activatingExamples}
                  onChangeText={(t) => setActivatingExamples(t.replace(/[^0-9]/g, ''))}
                  keyboardType="numeric"
                  placeholder="Optional (e.g. 25)"
                />
              </View>
              <View style={styles.inputField}>
                <Text style={styles.inputLabel}>Minimum activation value</Text>
                <TextInput
                  style={styles.numericInput}
                  value={minActivation}
                  onChangeText={(t) => setMinActivation(sanitizeDecimalInput(t))}
                  keyboardType="numeric"
                  placeholder="0.1"
                />
              </View>
              <View style={styles.inputField}>
                <Text style={styles.inputLabel}>Pagination</Text>
                <Text style={styles.loadingText}>Results are returned in pages of {pageSize} matches.</Text>
              </View>
            </View>

            <TouchableOpacity
              style={[styles.actionButton, styles.llmButton]}
              onPress={fetchLlmLabel}
              disabled={llmLoading}
            >
              <Text style={styles.actionButtonText}>
                {llmLoading ? 'Analyzing...' : `Analyze Feature #${feature?.id}`}
              </Text>
            </TouchableOpacity>

            {llmLoading && (
              <View style={styles.loadingRow}>
                <ActivityIndicator size="small" color="#2563eb" />
                <Text style={styles.loadingText}>Collecting activations and requesting LLM label...</Text>
              </View>
            )}

            {pageLoading && !llmLoading && (
              <View style={styles.loadingRow}>
                <ActivityIndicator size="small" color="#2563eb" />
                <Text style={styles.loadingText}>Loading page...</Text>
              </View>
            )}
          </View>

          <View style={styles.llmSection}>
            {llmError && <Text style={styles.llmError}>{llmError}</Text>}

            {llmLabel && (
              <View style={styles.llmResult}>
                <Text style={styles.resultTitle}>LLM Label Result</Text>
                <View style={styles.llmLabelRow}>
                  <Text style={styles.llmLabelText}>{llmLabel.label || '(no label returned)'}</Text>
                  <View
                    style={[
                      styles.confidenceBadge,
                      llmLabel.confidence === 'high'
                        ? styles.confHigh
                        : llmLabel.confidence === 'medium'
                          ? styles.confMed
                          : styles.confLow,
                    ]}
                  >
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

                {analysisMeta && (
                  <View style={styles.metaBox}>
                    <Text style={styles.metaText}>Dataset: {analysisMeta.dataset}</Text>
                    <Text style={styles.metaText}>Request ID: {analysisMeta.requestId || 'n/a'}</Text>
                    <Text style={styles.metaText}>LLM prompt activation mode: {analysisMeta.llmActivationMode}</Text>
                    <Text style={styles.metaText}>Sentences requested: {analysisMeta.sentenceLimit ?? 'auto'}</Text>
                    <Text style={styles.metaText}>Sentences scanned: {analysisMeta.scannedSentences}</Text>
                    <Text style={styles.metaText}>Activating examples requested: {analysisMeta.activatingExamplesLimit ?? 'all'}</Text>
                    <Text style={styles.metaText}>Min activation: {analysisMeta.minAct}</Text>
                    <Text style={styles.metaText}>Matches returned: {analysisMeta.totalMatches}</Text>
                    <Text style={styles.metaText}>Total pages: {analysisMeta.totalPages}</Text>
                    <Text style={styles.metaText}>Sentences used for LLM: {analysisMeta.corpusSentencesUsed}</Text>
                  </View>
                )}

                {llmLabel.llm_prompt_examples?.length > 0 && (
                  <View style={styles.metaBox}>
                    <Text style={styles.metaText}>Exact examples sent to LLM prompt:</Text>
                    {llmLabel.llm_prompt_examples.slice(0, 12).map((ex, i) => (
                      <Text key={`llm-ex-${i}`} style={styles.metaText}>
                        {`${i + 1}. [act=${Number(ex.activation || 0).toFixed(3)}] ${ex.context}`}
                      </Text>
                    ))}
                  </View>
                )}
              </View>
            )}
          </View>

          {(usePromptAlignedRows || activationPayload?.matches?.length > 0) && (
            <View style={styles.activationsListWrap}>
              <Text style={styles.subheading}>
                {usePromptAlignedRows ? 'Top Activation Matches (Prompt-Aligned)' : 'Top Activation Matches'}
              </Text>

              {usePromptAlignedRows ? (
                promptAlignedRows.map((row, idx) => (
                  <View key={row.id} style={styles.matchCard}>
                    <View style={styles.matchMetaRow}>
                      <Text style={styles.matchMetaText}>Prompt example #{idx + 1}</Text>
                      <Text style={styles.matchActivation}>{Number(row.activation || 0).toFixed(4)}</Text>
                    </View>
                    <Text style={styles.matchContext}>{row.context}</Text>
                  </View>
                ))
              ) : (
                activationPayload.matches.map((match, idx) => (
                  <View key={`match-${idx}`} style={styles.matchCard}>
                    <View style={styles.matchMetaRow}>
                      <Text style={styles.matchMetaText}>
                        Sentence #{match.sentence_index} | Token #{match.token_index}
                      </Text>
                      <Text style={styles.matchActivation}>{Number(match.activation || 0).toFixed(4)}</Text>
                    </View>

                    <Text style={styles.matchContext}>
                      {match.left_context}
                      <Text style={styles.matchToken}>{match.token}</Text>
                      {match.right_context}
                    </Text>

                    <Text style={styles.matchSentence}>{match.sentence}</Text>
                  </View>
                ))
              )}

            </View>
          )}
        </View>
      </ScrollView>

      {activationPayload?.matches?.length > 0 && !usePromptAlignedRows && (
        <View style={styles.fixedPaginationBar}>
          <TouchableOpacity
            style={[styles.doneButton, (!activationPayload?.has_prev_page || llmLoading || pageLoading) && { opacity: 0.5 }]}
            disabled={!activationPayload?.has_prev_page || llmLoading || pageLoading}
            onPress={async () => {
              if (!lastActivationQuery || !activationPayload?.has_prev_page) return;
              const prevPage = Math.max(1, matchesPage - 1);
              try {
                await loadActivationPage(lastActivationQuery, prevPage);
              } catch (e) {
                setLlmError(e.message);
              }
            }}
          >
            <Text style={styles.doneButtonText}>Prev</Text>
          </TouchableOpacity>

          <Text style={styles.metaText}>Page {activationPayload.page} / {activationPayload.total_pages}</Text>

          <TouchableOpacity
            style={[styles.doneButton, (!activationPayload?.has_next_page || llmLoading || pageLoading) && { opacity: 0.5 }]}
            disabled={!activationPayload?.has_next_page || llmLoading || pageLoading}
            onPress={async () => {
              if (!lastActivationQuery || !activationPayload?.has_next_page) return;
              const nextPage = matchesPage + 1;
              try {
                await loadActivationPage(lastActivationQuery, nextPage);
              } catch (e) {
                setLlmError(e.message);
              }
            }}
          >
            <Text style={styles.doneButtonText}>Next</Text>
          </TouchableOpacity>
        </View>
      )}
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
    justifyContent: 'flex-start',
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
  contentScroll: {
    flex: 1,
  },
  mainContent: {
    padding: 20,
    paddingBottom: 96,
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
    backgroundColor: '#eff6ff',
    padding: 10,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#bfdbfe',
    alignItems: 'flex-end',
  },
  modelInfo: { fontSize: 10, color: '#64748b' },
  modelDetail: { fontSize: 10, color: '#64748b' },
  modelIndex: { fontSize: 12, fontWeight: 'bold', color: '#334155' },

  analysisPanel: {
    backgroundColor: '#f8fafc',
    borderWidth: 1,
    borderColor: '#e2e8f0',
    borderRadius: 12,
    padding: 14,
    marginBottom: 16,
    gap: 10,
  },
  panelTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#0f172a',
  },
  panelHint: {
    color: '#475569',
    fontSize: 13,
    lineHeight: 20,
  },
  inputRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  inputField: {
    flex: 1,
    minWidth: 180,
  },
  inputLabel: {
    fontSize: 12,
    color: '#334155',
    marginBottom: 6,
    fontWeight: '600',
  },
  numericInput: {
    borderWidth: 1,
    borderColor: '#cbd5e1',
    borderRadius: 8,
    backgroundColor: '#fff',
    paddingVertical: 8,
    paddingHorizontal: 10,
    fontSize: 14,
    color: '#0f172a',
    outlineStyle: 'none',
  },
  loadingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  loadingText: {
    color: '#334155',
    fontSize: 13,
  },

  llmSection: {
    marginBottom: 20,
    gap: 10,
  },
  llmButton: {
    backgroundColor: '#2563eb',
    alignSelf: 'flex-start',
  },
  llmError: {
    color: '#dc2626',
    fontSize: 13,
  },
  llmResult: {
    backgroundColor: '#eff6ff',
    borderRadius: 8,
    padding: 14,
    borderWidth: 1,
    borderColor: '#bfdbfe',
    gap: 6,
  },
  resultTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: '#1e3a8a',
    marginBottom: 4,
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
    color: '#1e3a8a',
    flex: 1,
  },
  confidenceBadge: {
    borderRadius: 20,
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  confHigh: { backgroundColor: '#bbf7d0' },
  confMed: { backgroundColor: '#fef08a' },
  confLow: { backgroundColor: '#fecaca' },
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
    backgroundColor: '#dbeafe',
    borderRadius: 12,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderWidth: 1,
    borderColor: '#93c5fd',
  },
  chipText: {
    fontSize: 11,
    color: '#1e3a8a',
    fontWeight: '600',
  },
  metaBox: {
    marginTop: 8,
    backgroundColor: '#f8fafc',
    borderWidth: 1,
    borderColor: '#e2e8f0',
    borderRadius: 8,
    padding: 10,
    gap: 2,
  },
  metaText: {
    fontSize: 12,
    color: '#334155',
  },

  actionButton: {
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 8,
  },
  actionButtonText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 13,
  },

  activationsListWrap: {
    marginBottom: 20,
  },
  subheading: {
    fontSize: 16,
    fontWeight: '700',
    color: '#0f172a',
    marginBottom: 10,
  },
  matchCard: {
    backgroundColor: '#f8fafc',
    borderRadius: 10,
    padding: 12,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  matchMetaRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  matchMetaText: {
    fontSize: 12,
    color: '#334155',
    fontWeight: '600',
  },
  matchActivation: {
    fontSize: 12,
    color: '#0f766e',
    fontWeight: '700',
  },
  matchContext: {
    flex: 1,
    fontSize: 14,
    color: '#1f2937',
    lineHeight: 22,
    marginBottom: 6,
  },
  matchToken: {
    backgroundColor: '#d1fae5',
    color: '#065f46',
    fontWeight: '700',
  },
  matchSentence: {
    fontSize: 12,
    lineHeight: 18,
    color: '#64748b',
  },
  paginationRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: 8,
    marginBottom: 16,
    gap: 12,
  },
  fixedPaginationBar: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderTopWidth: 1,
    borderTopColor: '#e2e8f0',
    backgroundColor: '#ffffff',
  },
});
