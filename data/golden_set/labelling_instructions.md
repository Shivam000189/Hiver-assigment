# Customer Support Agent — Golden Evaluation Set Labelling Instructions

Welcome to the annotation task for the **AppleSupport AI Agent Golden Evaluation Set**.
This dataset of 200 customer interactions serves as the immutable ground-truth benchmark for evaluating intent classification, escalation gating, and reply generation.

---

## 1. The Annotation Task

For each row in `golden_label_sheet.csv`, provide four annotations:

1. **`intent`** *(Required)*: Exactly ONE intent name from the 9 taxonomy classes in Section 2.
2. **`escalate`** *(Required)*: `yes` or `no`. Does this inquiry require immediate escalation to a human supervisor/specialist team rather than automated resolution?
3. **`escalate_reason`** *(Required if escalate=yes, optional if no)*: A one-line explanation of the escalation trigger based on the 5 Escalation Rules in Section 3.
4. **`good_reply_elements`** *(Required)*: 1–3 concise bullet points defining the necessary criteria for an ideal agent reply to this specific customer.

---

## 2. Intent Taxonomy & Decision Rules

| Intent Class | Definition | Distinguishing Cue & Boundary | Examples |
| :--- | :--- | :--- | :--- |
| `keyboard_autocorrect_bug` | Customer reports abnormal keyboard behavior, specifically the infamous iOS autocorrect glitch where typing 'I' produces strange symbols (e.g. 'A [?]' or 'I️'), frozen keyboards, or predictive text corruption. | Focuses specifically on text input, letter replacements, and keyboard crashes. Distinguish from `system_performance_freeze` (general OS slowness). | - *"FIX THIS I️ SHIT"*<br>- *"I'm sick of not being able to type the word I"*<br>- *"Every time I type the letter i, I get this symbol"* |
| `system_performance_freeze` | Customer experiences device lockups, sluggish animation response, apps crashing upon opening, touch screen unresponsiveness, or unexpected restarts following a software update. | Device or OS-level performance degradation. Distinguish from `battery_drain_power` (battery loss is the primary symptom). | - *"Latest update has turned my phone into a stuttering mess"*<br>- *"Phone constantly freezes and apps crash"*<br>- *"iOS 11.1.2 has left my phone unusable"* |
| `battery_drain_power` | Customer complains about rapid battery percentage drop, unexpected device shutdowns at high percentages (e.g. dying at 30%), overheating while in use or charging, or failure to charge. | Explicit battery discharge, charging, or thermal complaints. | - *"iOS 11.0.2 drains my entire phone battery in minutes"*<br>- *"Battery on my iPhone 7 draining super fast since update"*<br>- *"Phone gets blazing hot while plugged in"* |
| `audio_music_playback` | Customer reports Apple Music playback stops, downloaded songs disappearing, headphone/adapter audio crackling, or Spotify app crashing during playback. | Audio playback, music library sync, headphones, sound bugs. Distinguish from `connectivity_network` (Bluetooth pairing protocol errors). | - *"Every time I listen to music and go to another app it stops playing"*<br>- *"Cancelled subscription and my downloaded songs are locked"*<br>- *"Audio crackling through lightning adapter"* |
| `connectivity_network` | Customer reports persistent Wi-Fi disconnects, failure to pair Bluetooth peripherals (AirPods, car stereo), AirDrop failures, or 'No Service' cellular data drops. | Hardware/protocol connectivity errors (Wi-Fi, Bluetooth, LTE, AirDrop). | - *"iPhone won't connect to my home Wi-Fi after update"*<br>- *"Bluetooth keeps dropping in my car"*<br>- *"AirDrop not finding nearby devices"* |
| `camera_photos_media` | Customer encounters a black screen in the Camera app, blurry focus, flashlight disabled due to temperature warnings, or photos failing to save to Camera Roll. | Camera sensor, lens, flashlight, or Photos library corruptions. | - *"Camera app opens to a completely black screen"*<br>- *"Flashlight disabled saying iPhone needs to cool down"*<br>- *"Photos took yesterday disappeared from camera roll"* |
| `account_icloud_login` | Customer is locked out of their Apple ID, fails to receive two-factor authentication verification codes, encounters password reset loops, or cannot sync iCloud storage backups. | Identity credentials, 2FA codes, account lockout, and iCloud backup tiers. Distinguish from `billing_app_store` (credit card billing/purchases). | - *"Locked out of my Apple ID and not receiving 2FA code"*<br>- *"Password reset email never arrives"*<br>- *"iCloud storage full but I deleted all my backups"* |
| `billing_app_store` | Customer reports unexpected credit card charges from Apple, struggles to cancel active subscriptions, encounters App Store download/update verification errors, or requests a refund. | Financial transactions, unauthorized charges, in-app purchases, App Store billing. | - *"Charged $9.99 for an app I deleted weeks ago"*<br>- *"How do I cancel my subscription and get a refund?"*<br>- *"App Store says payment method declined but card is active"* |
| `other` | General praise, design complaints, shipping inquiries for new devices, emoji feedback, or vague statements without a specific troubleshooting category. | Residual class (<15%). Use only when no clear technical category applies. | - *"Love the new emoji keyboard!"*<br>- *"When is the iPhone X shipping to UK stores?"*<br>- *"Why did you move the search bar?"* |

---

## 3. Brand Escalation Policy (5 Concrete Rules)

An inquiry MUST be marked **`escalate: yes`** if it triggers ANY of the following 5 criteria:

1. **Rule 1: Legal / Regulatory / Law Enforcement Threat**
   - Customer explicitly mentions hiring a lawyer, filing a lawsuit, contacting the FTC/consumer protection bureau, reporting fraud to the police, or initiating credit card chargebacks.
2. **Rule 2: Hardware Safety Hazard & Physical Danger**
   - Reports of battery swelling, smoking, burning, electrical shock, melting chargers, or devices exploding. Immediate safety escalation required.
3. **Rule 3: Severe Security Breach & Identity Theft**
   - Account takeover, unauthorized device linked to Apple ID, compromised credentials, or extortion/ransomware threats.
4. **Rule 4: Repeated Unresolved Failure & Severe Business Disruption**
   - Multi-turn interactions where standard troubleshooting (reboots, network resets, text replacements) has repeatedly failed (3+ attempts) or critical business/work device is completely inoperable.
5. **Rule 5: Abusive Communication & Explicit Harassment**
   - Extreme profanity directed at agents, threats of violence, or harassment necessitating manager intervention.

---

## 4. Worked Examples

### Example 1 (Easy Technical Diagnosis)
- **Customer Text**: *"My iPhone 7 battery is dying in 2 hours since updating to iOS 11.0.3. It used to last all day. Please fix!"*
- **`intent`**: `battery_drain_power`
- **`escalate`**: `no`
- **`escalate_reason`**: `N/A - Standard troubleshooting applies.`
- **`good_reply_elements`**:
  - Acknowledge iOS 11.0.3 battery drain issue.
  - Advise checking `Settings > Battery` for power-draining apps.
  - Offer DM link to run automated battery diagnostics.

### Example 2 (Easy Account Query)
- **Customer Text**: *"I forgot my Apple ID password and my trusted phone number changed. How do I get into my account?"*
- **`intent`**: `account_icloud_login`
- **`escalate`**: `no`
- **`escalate_reason`**: `N/A - Standard account recovery path.`
- **`good_reply_elements`**:
  - Direct to official recovery portal `iforgot.apple.com`.
  - Explain Account Recovery process for changed phone numbers.
  - Caution against sharing credentials over public Twitter.

### Example 3 (Ambiguous Multi-Issue Query)
- **Customer Text**: *"Since the update my phone lags horribly when opening Apple Music and then the song cuts out when I lock the screen. What is going on?"*
- **`intent`**: `audio_music_playback` (Dominant issue is the media playback disruption)
- **`escalate`**: `no`
- **`escalate_reason`**: `N/A - Non-critical software glitch.`
- **`good_reply_elements`**:
  - Request iOS version and Music app status.
  - Suggest toggling iCloud Music Library and force-restarting device.
  - Ask if issue occurs with downloaded vs streaming tracks.

### Example 4 (Ambiguous Sarcasm / Frustration)
- **Customer Text**: *"Thanks Apple for the amazing update that turned my $1000 iPhone 8 into an overpriced brick that won't connect to Wi-Fi!"*
- **`intent`**: `connectivity_network`
- **`escalate`**: `no`
- **`escalate_reason`**: `N/A - Sarcastic tone but standard Wi-Fi troubleshooting.`
- **`good_reply_elements`**:
  - De-escalate with empathetic tone.
  - Provide network reset instructions (`Settings > General > Reset > Reset Network Settings`).
  - Request DM if Wi-Fi toggle remains greyed out.

### Example 5 (Escalation: Safety / Battery Swelling)
- **Customer Text**: *"My iPhone battery expanded and cracked the screen open while charging overnight! The back is burning hot!"*
- **`intent`**: `battery_drain_power`
- **`escalate`**: `yes`
- **`escalate_reason`**: `Rule 2: Hardware Safety Hazard (battery swelling/overheating).`
- **`good_reply_elements`**:
  - Prioritize safety: instruct customer to immediately unplug and stop using the device.
  - Direct to urgent priority DM and schedule immediate Apple Store safety evaluation.

### Example 6 (Escalation: Legal & Financial Threat)
- **Customer Text**: *"You charged my card $400 for unauthorized subscriptions. I've contacted my lawyer and am filing fraud charges with my bank and police today!"*
- **`intent`**: `billing_app_store`
- **`escalate`**: `yes`
- **`escalate_reason`**: `Rule 1: Legal threat and criminal fraud allegation.`
- **`good_reply_elements`**:
  - Acknowledge severity of unauthorized billing claim immediately.
  - Direct to senior billing specialist via secure DM channel.
  - Guide to `reportaproblem.apple.com` while routing to supervisor queue.

---

## 5. Edge-Case Annotation Policies

1. **Multi-Issue Inquiries**: Identify the primary/dominant technical blocker for `intent`. In `good_reply_elements`, include resolution steps for both issues.
2. **Short Fragmentary Tweets** (e.g., *"I did that already"*): Check the `brand_reply_text` or `followup_text` column for context. If context reveals the underlying issue (e.g. battery), label accordingly; otherwise classify as `other`.
3. **Non-English / Emoji-Only Tweets**: If English intent is unidentifiable, classify as `other` and `escalate: no`.
4. **Sarcasm**: Classify by the actual technical problem being criticized, not the literal positive phrasing.
