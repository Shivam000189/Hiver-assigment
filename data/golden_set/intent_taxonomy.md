# AppleSupport Customer Support Intent Taxonomy

This taxonomy defines **9 grounded, actionable customer support intents** mined from empirical clustering and semantic analysis of the `AppleSupport` conversation dataset (~102k unique customer queries).

---

## Intent: `keyboard_autocorrect_bug`

- **Title**: Keyboard Glitches & Autocorrect Bugs
- **Definition**: Customer reports abnormal keyboard behavior, specifically the infamous iOS autocorrect glitch where typing the letter 'I' produces strange symbols (e.g., 'A [?]' or 'I️'), frozen keyboards, or predictive text corruption.
- **Distinguish from**: Distinguish from `system_performance_freeze` (which covers general OS slowdowns/app crashes rather than specific keyboard/typing malfunctions).
- **Prevalence**: ~22.0% of labelled sample (22.1% of full corpus)
- **Examples**:
  1. "I️ don’t even have to type anything to get the #predictivetext #iosglitch @AppleSupport @34173 #apple @175600 https://t.co/p8IqzbDyD0"
  2. "@115858 Fix the i’s. This shit annoying."
  3. "@AppleSupport I can’t make calls. It just keeps ending the call before it’s even rung. WTF."
- **Typical resolution pattern**: AppleSupport acknowledges the known keyboard autocorrect bug, points the customer to the official workaround in Settings > General > Keyboard > Text Replacement (e.g. shortcut 'I' for 'i'), and advises updating to the latest iOS patch.

---

## Intent: `system_performance_freeze`

- **Title**: System Performance, Freezing & iOS Updates
- **Definition**: Customer experiences device lockups, sluggish animation response, apps crashing upon opening, touch screen unresponsiveness, or unexpected restarts following a system software update.
- **Distinguish from**: Distinguish from `battery_drain_power` (where power loss is the primary issue) and `keyboard_autocorrect_bug` (isolated text input glitches).
- **Prevalence**: ~43.7% of labelled sample (45.7% of full corpus)
- **Examples**:
  1. "@AppleSupport On 11.0.2. Planning to update tonight."
  2. "@AppleSupport I phone 7 with 11.0.3"
  3. "@AppleSupport Tried that. No different i’m afraid"
- **Typical resolution pattern**: AppleSupport asks for the exact iOS build version and device model, recommends performing a force restart, and requests a Direct Message (DM) to troubleshoot background processes.

---

## Intent: `battery_drain_power`

- **Title**: Battery Drain, Overheating & Power Issues
- **Definition**: Customer complains about rapid battery percentage drop, unexpected device shutdowns at high percentages (e.g., dying at 30%), overheating while in use or charging, or failure to charge.
- **Distinguish from**: Distinguish from `system_performance_freeze` (general OS slowness without explicit power/battery degradation complaints).
- **Prevalence**: ~13.2% of labelled sample (11.1% of full corpus)
- **Examples**:
  1. "So... @115858 still hasnt charged me for my phone, despite confirming my order by email and phone several days ago. What up with that?"
  2. "@AppleSupport My MacBook Pro just went blank and won’t turn on. It’s fully charged😫😫🙁"
  3. "@AppleSupport hi Apple. I bought a new iPhone 7 yesterday! It already had iOS 11 on it. I do feel however like my battery is dropping quickly despite it being new! What do I do?"
- **Typical resolution pattern**: AppleSupport prompts the user to check Battery Usage under Settings > Battery, verifies if the drain occurs on Wi-Fi or cellular, and directs the user to an automated battery diagnostic link via DM.

---

## Intent: `audio_music_playback`

- **Title**: Apple Music, Audio & Media Playback
- **Definition**: Customer reports Apple Music playback stops, downloaded songs disappearing, headphone/adapter audio crackling, or Spotify/music app crashing during playback.
- **Distinguish from**: Distinguish from `connectivity_network` (Bluetooth pairing protocol failures) and `system_performance_freeze` (general app crashes unrelated to audio).
- **Prevalence**: ~4.8% of labelled sample (3.9% of full corpus)
- **Examples**:
  1. "@515161 @AppleSupport So lame😡 me too 😡"
  2. "@115858 just got new headphones with the iPhone 8 and they started making a weird noise but I kept them in anyway. They just POPPED in my ear and now my ear hurts, what are you going to do about this?"
  3. "Spent $170 on @115858 AirPods at @127271 not even a month later the left one quits working. Nice to know something so expensive that doesn’t come with an extended warranty craps out so quick. @AppleSupport @138244"
- **Typical resolution pattern**: AppleSupport checks if playback errors happen across downloaded vs. streamed songs, advises toggling iCloud Music Library / Sync Library, and suggests reinstalling the Music app.

---

## Intent: `connectivity_network`

- **Title**: Wi-Fi, Bluetooth & Cellular Connectivity
- **Definition**: Customer reports persistent Wi-Fi disconnects, failure to pair Bluetooth peripherals (AirPods, car stereo), AirDrop failures, or 'No Service' cellular data drops.
- **Distinguish from**: Distinguish from `audio_music_playback` (audio streaming buffer errors) and `system_performance_freeze` (system UI freezing).
- **Prevalence**: ~2.3% of labelled sample (1.4% of full corpus)
- **Examples**:
  1. "@AppleSupport the music on my iPhone keeps pausing whilst I’m driving in my car it’s via Bluetooth. It’s only happened since the update."
  2. "@170666 I keep losing data connection. Cellular is fine, just data. @AppleSupport why is this happening?"
  3. "Ever since the most recent iOS update my Bluetooth works about 30% as well, phone freezes and generally sucks. Please fix @115858"
- **Typical resolution pattern**: AppleSupport suggests resetting Network Settings (`Settings > General > Reset > Reset Network Settings`), toggling Airplane Mode, and testing on an alternate Wi-Fi network before escalating via DM.

---

## Intent: `camera_photos_media`

- **Title**: Camera, Flashlight & Photos App Glitches
- **Definition**: Customer encounters a black screen in the Camera app, blurry focus, flashlight disabled due to temperature warnings, or photos failing to save to Camera Roll.
- **Distinguish from**: Distinguish from `system_performance_freeze` (general app launch freezes) and physical hardware cracked lenses.
- **Prevalence**: ~2.2% of labelled sample (0.6% of full corpus)
- **Examples**:
  1. "@AppleSupport my FaceTime Live Photo’s are not saving"
  2. "@AppleSupport hi ever since I updated my phone to ios11 my rear camera hasn't worked properly please help"
  3. "I’ve read so many great reviews for the #iPhoneX’s camera and man I am severely disappointed. I hope I just got a defective unit cause this is worse than my 7+ massively. @AppleSupport"
- **Typical resolution pattern**: AppleSupport asks if the camera issue occurs across both front and rear lenses as well as in third-party apps, recommends closing all camera-related apps, and requests a DM.

---

## Intent: `account_icloud_login`

- **Title**: Apple ID, iCloud, 2FA & Account Access
- **Definition**: Customer is locked out of their Apple ID, fails to receive two-factor authentication verification codes, encounters password reset loops, or cannot sync iCloud storage backups.
- **Distinguish from**: Distinguish from `billing_app_store` (which focuses on charges and refunds rather than authentication and identity security).
- **Prevalence**: ~2.7% of labelled sample (1.9% of full corpus)
- **Examples**:
  1. "@AppleSupport how can I remove a file I’ve downloaded within Files locally to my iPhone X and leave it in iCloud Drive without deleting the file entirely?"
  2. "@115858 I need help with my Apple ID"
  3. "@AppleSupport Oh now it wants to sign me out my fucking iCloud dude!!!!!!!! I am literallly having a BF"
- **Typical resolution pattern**: AppleSupport provides direct links to iforgot.apple.com, guides the user through two-factor authentication account recovery protocols, and offers account security verification guidance.

---

## Intent: `billing_app_store`

- **Title**: App Store, In-App Purchases & Subscriptions
- **Definition**: Customer reports unexpected credit card charges from Apple, struggles to cancel active subscriptions, encounters App Store download/update verification errors, or requests a refund.
- **Distinguish from**: Distinguish from `account_icloud_login` (account credentials and storage tiers vs transaction billing).
- **Prevalence**: ~2.2% of labelled sample (1.0% of full corpus)
- **Examples**:
  1. "Hello @AppleSupport - Why macOS Sierra no longer appears in the “my purchases” section ?"
  2. "@AppleSupport Hi, I downloaded a few text tones from the App Store, but I can’t seem to find them on my phone to actually set them as my text tone. Can you help?"
  3. "@AppleSupport Old Film purchasers disappeared from my account. Please help."
- **Typical resolution pattern**: AppleSupport directs the customer to reportaproblem.apple.com to inspect their purchase history, cancel active subscriptions, and submit formal refund requests.

---

## Intent: `other`

- **Title**: General Feedback, Praise & Ambiguous Inquiries
- **Definition**: Customer sends general brand feedback, praise, emojis, complaints without actionable technical context, or brief follow-up replies.
- **Distinguish from**: Used exclusively when the inquiry does not fall into any of the 8 structured technical support categories.
- **Prevalence**: ~7.0% of labelled sample (12.4% of full corpus)
- **Examples**:
  1. "@115858 It's saying it can't connect to the server."
  2. "@AppleSupport my app suddenly is not working https://t.co/VkbBZsH8bW"
  3. "Only to break File Sharing! Sigh :-( /cc @115858 https://t.co/QGw2mX4igg"
- **Typical resolution pattern**: AppleSupport provides a polite acknowledgment, asks for clarifying details regarding the issue, or directs general product suggestions to apple.com/feedback.

---

