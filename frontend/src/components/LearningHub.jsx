import React, { useMemo, useState } from 'react';

const CASE_LIBRARY = [
  {
    id: 'case-1',
    bone: 'Wrist',
    age: '42',
    difficulty: 'Intermediate',
    fractureType: 'Spiral fracture',
    mechanism: 'Fall from a ladder',
    symptoms: 'Wrist pain and swelling',
    history: 'Fall from a ladder with wrist pain and swelling.',
    diagnosis: 'Distal radius spiral fracture with mild displacement.',
    explanation: 'Spiral fracture lines wrap around the shaft, which matches a twisting injury mechanism.',
    learningPoints: ['Look for oblique fracture lines that rotate around the cortex.', 'Compare AP and lateral views to assess displacement.', 'Use mechanism of injury to separate spiral from oblique patterns.'],
    quality: 'Good',
    views: ['AP', 'Lateral', 'Oblique'],
    image: 'Wrist AP and lateral radiographs with a visible distal radius fracture.',
  },
  {
    id: 'case-2',
    bone: 'Tibia',
    age: '8',
    difficulty: 'Beginner',
    fractureType: 'Greenstick fracture',
    mechanism: 'Jumped from a low wall',
    symptoms: 'Limping and localized pain',
    history: 'Jumped from a low wall and landed awkwardly.',
    diagnosis: 'Incomplete greenstick fracture of the mid-tibia.',
    explanation: 'Pediatric bones bend before breaking, so one cortex fails while the other remains intact.',
    learningPoints: ['Pediatric bone flexibility changes fracture appearance.', 'Check the intact cortex before calling a complete fracture.', 'Use the contralateral side when anatomy is still developing.'],
    quality: 'Excellent',
    views: ['AP', 'Lateral'],
    image: 'Pediatric tibia radiograph showing cortical bowing and incomplete break.',
  },
  {
    id: 'case-3',
    bone: 'Spine',
    age: '67',
    difficulty: 'Advanced',
    fractureType: 'Compression fracture',
    mechanism: 'Lifting a heavy box',
    symptoms: 'Sudden back pain',
    history: 'Sudden back pain after lifting a heavy box.',
    diagnosis: 'Vertebral compression fracture with reduced anterior height.',
    explanation: 'Collapsed anterior vertebral height and wedge shape are classic compression findings.',
    learningPoints: ['Assess vertebral height ratios to judge severity.', 'Consider osteoporosis or trauma based on age and mechanism.', 'Measure Cobb angle when alignment is affected.'],
    quality: 'Moderate',
    views: ['Lateral', 'Oblique'],
    image: 'Lateral spine image with wedge deformity and reduced vertebral height.',
  },
  {
    id: 'case-4',
    bone: 'Foot',
    age: '29',
    difficulty: 'Intermediate',
    fractureType: 'Stress fracture',
    mechanism: 'Increased running load',
    symptoms: 'Persistent foot pain',
    history: 'Runner with persistent foot pain after training increase.',
    diagnosis: 'Metatarsal stress fracture with subtle cortical reaction.',
    explanation: 'Stress fractures may appear faint, so line visibility plus localized pain are both important.',
    learningPoints: ['Image quality matters when the fracture line is subtle.', 'Symptom persistence can be more telling than a dramatic x-ray finding.', 'Combine clinical correlation with multiple views.'],
    quality: 'Fair',
    views: ['AP', 'Oblique'],
    image: 'Metatarsal image with a faint cortical irregularity and periosteal reaction.',
  },
];

const DIFF_OPTIONS = [
  'Fracture',
  'Osteoarthritis',
  'Osteomyelitis',
  'Bone tumor',
  'Bone cyst',
  'Normal variant',
];

const TUTOR_HINTS = {
  'Why is this a spiral fracture?': 'A spiral fracture usually forms when the bone is twisted, creating a helical fracture line that wraps around the cortex.',
  'Why not an oblique fracture?': 'Oblique fractures have a more straight diagonal line, while spiral fractures curve around the shaft and better match a twisting force.',
  'Which view is best?': 'Use the view that best shows the fracture plane; lateral is often helpful for displacement, while AP and oblique can clarify cortical edges.',
};

const PROGRESS_SAMPLE = [
  { label: 'Cases completed', value: 28, max: 40 },
  { label: 'Accuracy', value: 86, max: 100 },
  { label: 'Weak region mastery', value: 61, max: 100 },
  { label: 'Improvement', value: 14, max: 20 },
];

function LearningHub() {
  const [selectedCaseId, setSelectedCaseId] = useState(CASE_LIBRARY[0].id);
  const [annotationNote, setAnnotationNote] = useState('');
  const [selectedQuestion, setSelectedQuestion] = useState('Why is this a spiral fracture?');
  const [multiViewEnabled, setMultiViewEnabled] = useState(['AP', 'Lateral']);
  const [selectedQuality, setSelectedQuality] = useState('Motion blur');
  const [measurementMode, setMeasurementMode] = useState('Fracture displacement');

  const selectedCase = useMemo(
    () => CASE_LIBRARY.find((item) => item.id === selectedCaseId) || CASE_LIBRARY[0],
    [selectedCaseId],
  );

  const tutorAnswer = TUTOR_HINTS[selectedQuestion] || 'Ask a question about the image pattern, view selection, or fracture logic to get a teaching explanation.';

  const differentialRank = useMemo(() => {
    return DIFF_OPTIONS.map((label) => {
      const scoreMap = {
        Fracture: 96,
        Osteoarthritis: 22,
        Osteomyelitis: 35,
        'Bone tumor': 18,
        'Bone cyst': 12,
        'Normal variant': 28,
      };

      const rationaleMap = {
        Fracture: 'Best fit when there is a clear cortical break, trauma, or localized tenderness.',
        Osteoarthritis: 'More likely when joint-space narrowing and osteophytes dominate instead of a fracture line.',
        Osteomyelitis: 'Consider if there is fever, swelling, and bone destruction rather than a clean fracture pattern.',
        'Bone tumor': 'Think about this when there is a destructive lesion, expansile bone change, or atypical pain pattern.',
        'Bone cyst': 'Usually shows a lytic lesion with more rounded margins than an acute fracture.',
        'Normal variant': 'Possible if the line is smooth, symmetric, or matches a growth or anatomical variant.',
      };

      return {
        label,
        score: scoreMap[label],
        rationale: rationaleMap[label],
      };
    }).sort((a, b) => b.score - a.score);
  }, []);

  return (
    <div className="learning-hub page-shell">
      <header className="learning-hub__hero">
        <p className="opening-kicker">Educational workspace</p>
        <h1 className="learning-hub__title">Practice reading fractures with guided feedback</h1>
        <p className="learning-hub__subtitle">
          Explore a case library, practice annotations, compare diagnoses, and review the same study through a clinical and teaching lens.
        </p>
      </header>

      <section className="learning-hub__grid learning-hub__grid--featured">
        <article className="learning-panel learning-panel--case">
          <div className="learning-panel__header">
            <div>
              <p className="learning-panel__eyebrow">Case library</p>
              <h2>Curated radiology examples</h2>
            </div>
            <div className="chip-row">
              {['Bone', 'Age', 'Difficulty', 'Fracture type'].map((label) => (
                <span key={label} className="chip">{label}</span>
              ))}
            </div>
          </div>

          <div className="case-selector">
            {CASE_LIBRARY.map((item) => (
              <button
                type="button"
                key={item.id}
                className={`case-selector__button ${selectedCaseId === item.id ? 'is-active' : ''}`}
                onClick={() => setSelectedCaseId(item.id)}
              >
                <strong>{item.bone}</strong>
                <span>{item.fractureType}</span>
              </button>
            ))}
          </div>

          <div className="case-card">
            <div className="case-card__header">
              <div>
                <p className="learning-panel__eyebrow">Selected case</p>
                <h3>{selectedCase.bone} fracture case</h3>
              </div>
              <div className="case-meta">
                <span>{selectedCase.age} years old</span>
                <span>{selectedCase.difficulty}</span>
                <span>{selectedCase.quality} image quality</span>
              </div>
            </div>

            <div className="case-card__image">
              {selectedCase.image}
            </div>

            <div className="case-card__body">
              <div>
                <h4>Clinical history</h4>
                <p>{selectedCase.history}</p>
              </div>
              <div>
                <h4>Diagnosis</h4>
                <p>{selectedCase.diagnosis}</p>
              </div>
              <div>
                <h4>Explanation</h4>
                <p>{selectedCase.explanation}</p>
              </div>
              <div>
                <h4>Learning points</h4>
                <ul>
                  {selectedCase.learningPoints.map((point) => <li key={point}>{point}</li>)}
                </ul>
              </div>
            </div>
          </div>
        </article>

        <article className="learning-panel">
          <div className="learning-panel__header">
            <div>
              <p className="learning-panel__eyebrow">Annotation practice</p>
              <h2>Draw, mark, and compare</h2>
            </div>
          </div>
          <p className="learning-panel__text">
            Use the freeform note area below to simulate fracture lines, abnormal regions, and short teaching notes.
          </p>
          <textarea
            className="learning-textarea"
            value={annotationNote}
            onChange={(e) => setAnnotationNote(e.target.value)}
            placeholder="Example: mark the distal cortex break, note swelling, and track the fracture line across the AP view."
            rows={7}
          />
          <div className="learning-note-box">
            <strong>Expert comparison</strong>
            <p>
              {annotationNote
                ? `Student note captured. Compare your annotations against expert references for ${selectedCase.fractureType.toLowerCase()}.`
                : 'Add a note to compare your interpretation against the reference annotations.'}
            </p>
          </div>

          <div className="quality-checker">
            <div className="quality-checker__header">
              <p className="learning-panel__eyebrow">Image quality checker</p>
              <h3>Is this scan suitable for reading?</h3>
            </div>
            <div className="quality-options">
              {['Underexposure', 'Overexposure', 'Motion blur', 'Rotation', 'Missing anatomy'].map((item) => (
                <button key={item} type="button" className={`quality-pill ${selectedQuality === item ? 'is-active' : ''}`} onClick={() => setSelectedQuality(item)}>
                  {item}
                </button>
              ))}
            </div>
            <p className="learning-panel__text">Current quality signal: {selectedQuality}. Teach students whether the image is ready for interpretation or needs a repeat view.</p>
          </div>
        </article>
      </section>

      <section className="learning-hub__grid learning-hub__grid--three">
        <article className="learning-panel">
          <div className="learning-panel__header">
            <div>
              <p className="learning-panel__eyebrow">Differential diagnosis</p>
              <h2>Why this case is or is not a fracture</h2>
            </div>
          </div>
          <div className="differential-list">
            {differentialRank.map((item) => (
              <div key={item.label} className="differential-item">
                <div className="differential-item__top">
                  <strong>{item.label}</strong>
                  <span>{item.score}%</span>
                </div>
                <div className="differential-bar"><span style={{ width: `${item.score}%` }} /></div>
                <p>{item.rationale}</p>
              </div>
            ))}
          </div>
        </article>

        <article className="learning-panel">
          <div className="learning-panel__header">
            <div>
              <p className="learning-panel__eyebrow">Clinical correlation</p>
              <h2>Case context matters</h2>
            </div>
          </div>
          <div className="clinical-card">
            <div><span>Age</span><strong>{selectedCase.age}</strong></div>
            <div><span>Mechanism</span><strong>{selectedCase.mechanism}</strong></div>
            <div><span>Symptoms</span><strong>{selectedCase.symptoms}</strong></div>
          </div>
          <p className="learning-panel__text">Students interpret the X-ray together with age, trauma mechanism, and symptoms to improve reasoning accuracy.</p>

          <div className="learning-panel__subsection">
            <p className="learning-panel__eyebrow">Measurement tools</p>
            <div className="measurement-grid">
              {['Cobb angle', 'Bone length', 'Fracture displacement', 'Joint space', 'Calibration ruler'].map((tool, index) => (
                <div key={tool} className="measurement-chip">
                  <span>{tool}</span>
                  <strong>
                    {index === 0 ? '18°' : index === 1 ? '14.6 cm' : index === 2 ? '4 mm' : index === 3 ? '2.1 mm' : 'On'}
                  </strong>
                </div>
              ))}
            </div>
          </div>
        </article>

        <article className="learning-panel">
          <div className="learning-panel__header">
            <div>
              <p className="learning-panel__eyebrow">AI tutor chat</p>
              <h2>Ask a teaching question</h2>
            </div>
          </div>
          <div className="question-row">
            {Object.keys(TUTOR_HINTS).map((question) => (
              <button key={question} type="button" className={`question-chip ${selectedQuestion === question ? 'is-active' : ''}`} onClick={() => setSelectedQuestion(question)}>
                {question}
              </button>
            ))}
          </div>
          <div className="tutor-answer">
            <strong>AI tutor response</strong>
            <p>{tutorAnswer}</p>
          </div>

          <div className="learning-panel__subsection">
            <p className="learning-panel__eyebrow">Multi-view analysis</p>
            <div className="checkbox-grid">
              {['AP', 'Lateral', 'Oblique'].map((view) => (
                <button
                  key={view}
                  type="button"
                  className={`view-chip ${multiViewEnabled.includes(view) ? 'is-active' : ''}`}
                  onClick={() => setMultiViewEnabled((current) => current.includes(view) ? current.filter((item) => item !== view) : [...current, view])}
                >
                  {view}
                </button>
              ))}
            </div>
            <p className="learning-panel__text">Selected views: {multiViewEnabled.join(', ') || 'None'}.</p>
          </div>
        </article>
      </section>

      <section className="learning-hub__grid learning-hub__grid--wide">
        <article className="learning-panel">
          <div className="learning-panel__header">
            <div>
              <p className="learning-panel__eyebrow">Progress tracking</p>
              <h2>Learning progress and achievements</h2>
            </div>
          </div>

          <div className="progress-grid">
            {PROGRESS_SAMPLE.map((item) => (
              <div key={item.label} className="progress-card">
                <div className="progress-card__top">
                  <strong>{item.label}</strong>
                  <span>{item.value}/{item.max}</span>
                </div>
                <div className="progress-track">
                  <span style={{ width: `${(item.value / item.max) * 100}%` }} />
                </div>
              </div>
            ))}
          </div>

          <div className="badge-row">
            {['Case reviewer', 'Annotation pro', 'View master', 'Research ready'].map((badge) => (
              <span key={badge} className="badge-pill">{badge}</span>
            ))}
          </div>
        </article>

        <article className="learning-panel">
          <div className="learning-panel__header">
            <div>
              <p className="learning-panel__eyebrow">Research dashboard</p>
              <h2>Academic performance metrics</h2>
            </div>
          </div>

          <div className="research-grid">
            <div className="research-card"><span>Precision</span><strong>0.92</strong></div>
            <div className="research-card"><span>Recall</span><strong>0.89</strong></div>
            <div className="research-card"><span>Model confidence</span><strong>94%</strong></div>
            <div className="research-card"><span>Misclassifications</span><strong>8</strong></div>
          </div>

          <div className="confusion-matrix">
            <div>TP 84</div>
            <div>FP 7</div>
            <div>FN 10</div>
            <div>TN 91</div>
          </div>

          <p className="learning-panel__text">
            Useful for academic projects, this view surfaces performance trends and gives a place to review misclassified examples.
          </p>
        </article>
      </section>
    </div>
  );
}

export default LearningHub;