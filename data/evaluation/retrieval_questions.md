Q1 (Conceptual): What is the difference between “AI for Security” and “Security for AI”?

Reference answer: “AI for Security” is using AI technologies to solve security problems, or building AI into security products. “Security for AI” is designing and implementing security measures to protect AI technologies and infrastructure, which matters most when the AI is critical to business or nation-state infrastructure.
Evidence (p. 2, II.B): “‘AI for Security’ means how can we use AI technologies to solve security problems or use AI technology in security products or solutions.”
Evidence (p. 3, II.C): “‘Security for AI’ means designing, implementing security measures for AI technologies, since every technology has limitations, vulnerabilities and prone to attacks so it is imperative to secure AI infrastructure especially if its crucial to the business and nation state infrastructure.”
Note: The two evidence chunks sit on different pages, so this tests multi-chunk retrieval.

Q2 (Conceptual): Why does the author say Strong AI is not yet practical, and how does it differ from Weak AI?

Reference answer: Weak (Narrow) AI is trained for limited tasks and is already in use (Alexa, Siri). Strong AI would be artificial general intelligence that thinks like humans, with self-aware consciousness. It is still theoretical, with no real-world use cases, and researchers are only exploring it.
Evidence (p. 2, II.A): “Weak AI alternatively called Narrow AI, which has been trained to perform limited tasks.”
Evidence (p. 2, II.A): “Strong AI and its branches are theoretical concepts at the moment and not practical , do not have real world use cases at the moment. Researchers are currently exploring this possibility [8].”

Q3 (Keyword-heavy): What is MITRE ATLAS and what is its purpose?

Reference answer: MITRE ATLAS (Adversarial Threat Landscape for Artificial-intelligence Systems) is a framework built by MITRE, a not-for-profit security organization, specifically for AI. It raises awareness of evolving vulnerabilities and of the tactics, techniques and procedures for tackling them.
Evidence (p. 4, end of II.C.3): “MITRE, a prominent and not for profit security, has built a framework specifically for AI called MITRE ATLAS (Adversarial Threat Landscape for Artificial-intelligence Systems) to raise awareness of the evolving vulnerabilities and tactics, techniques and procedures to tackle them [22].”
Also relevant: reference [22] on p. 6 (a weak match, useful as a distractor check).

Q4 (Keyword-heavy): How do data poisoning, model poisoning, and model inversion attacks differ?

Reference answer:
Data poisoning: the attacker spams data storage or inserts mislabeled or irrelevant data, so predictions become inaccurate and unreliable.
Model poisoning: malicious data is introduced into the training data in a development or testing environment, which can shift the model’s behavior significantly. The paper contrasts this with adversarial attacks, which target models in production.
Model inversion: the attacker reverse-engineers the model to expose the data set it used, giving access to PII/MNPI, and these attacks are hard to detect.
Evidence (p. 4, II.C.3, items 2-4):
“An attacker can intentionally spam data storage, insert incorrectly labeled data sets, or irrelevant data sets deviating from the median range of the training set, resulting in inaccurate predictions, insecure, and unreliable outcomes.”
“Adversarial attacks target AI models or systems in their production environment, whereas model poisoning attacks target AI models in a development or testing environment.”
“An attacker can reverse engineer the model, thus exposing the data set being used, resulting in access to PII/MNPI”

Q5 (Architecture / framework structure): What are the four functions of the NIST AI RMF Core, and what does each one do?

Reference answer:
GOVERN: ensures strong governance that drives internal practices and norms and builds an organizational risk culture.
MAP: establishes the context to frame risks related to an AI system.
MEASURE: uses quantitative, qualitative or mixed-method tools to analyze, assess, benchmark and monitor AI risks.
MANAGE: allocates risk resources to mapped and measured risks on a regular basis, as defined by GOVERN.
Evidence (p. 5, bullet list): the four bullets beginning “GOVERN ensures strong governance...”, “The MAP function establishes the context...”, “The MEASURE function employs...”, “The MANAGE function allocates risk resources...”
Note: The list is introduced by “As seen in [27, Fig.3] there are following sections in the AI RMF Core:”, so a good chunker keeps the intro and all four bullets together.

Q6 (Figure/caption): What does Figure 2 show, and how are Weak AI and Strong AI broken down?

Reference answer: Figure 2, captioned “Weak AI vs Strong AI,” is a tree diagram with AI at the top. It splits into Strong AI (Artificial General Intelligence and Artificial Super Intelligence) and Weak AI (Narrow AI).
Evidence (p. 2): caption “FIGURE 2. Weak AI vs Strong AI” plus the diagram labels. The surrounding II.A text corroborates the Weak/Narrow and Strong/AGI relationship.
Note: The sub-type breakdown exists only in the image. A text-only extraction can match just the caption and the II.A prose, so score this question separately for “found the caption” and “answered the sub-type detail.”

Q7 (Specific factual): Which real-world example does the paper use to illustrate gender bias in AI, and in what year did it occur?

Reference answer: Amazon’s AI-based candidate screening, in 2018. A gender bias was reportedly found in its review of applicants.
Evidence (p. 3, II.C.1, item 4): “...for example, the Amazon applicant scandal in 2018, where AI was used to screen candidates, apparently, there was a gender bias found based on the review [21].”

Q8 (Specific factual): Since when has Qrator Labs used a threat detection model against DDoS attacks, and how does it work?

Reference answer: Since 2006. The model learns patterns from the visitors using the website and uses them to filter DDoS attacks with very high accuracy.
Evidence (p. 3, II.B use cases): “Since 2006, Qrator labs have been using a threat detection model to protect themselves against Distributed denial of service (DDOS) attacks based on the pattern learned from the visitors using the website, which model allows filtering attacks with a very high accuracy [13].”