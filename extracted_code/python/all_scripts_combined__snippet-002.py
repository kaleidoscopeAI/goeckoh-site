class AbaEngine:
    """Expanded ABA Therapeutics: Positive reinforcement, social stories, skill-building."""

    settings: SystemSettings
    voice_profile: VoiceProfile
    guidance_logger: GuidanceLogger
    progress: Dict[str, Dict[str, AbaProgress]] = field(default_factory=lambda: {
        cat: {skill: AbaProgress() for skill in skills}
        for cat, skills in ABA_SKILLS.items()
    })
    social_story_templates: Dict[str, str] = field(default_factory=lambda: {
        "transition": "Today, we're going from {current} to {next}. First, we say goodbye to {current}. Then, we walk calmly to {next}. It's okay to feel a little worried, but we'll have fun there!",
        "meltdown": "Sometimes we feel overwhelmed, like a storm inside. When that happens, we can take deep breaths: in for 4, out for 6. Or hug our favorite toy. Soon the storm passes, and we feel better.",
        "social": "When we see a friend, we can say 'Hi, want to play?' If they say yes, we share the toys. If no, that's okay – we can play next time."
    })
    strategy_advisor: StrategyAdvisor = field(default_factory=StrategyAdvisor)

    def __post_init__(self) -> None:
        self.load_progress()

    def load_progress(self) -> None:
        """Load from guidance_events.csv if exists."""
        if self.settings.paths.guidance_csv.exists():
            with self.settings.paths.guidance_csv.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["category"] == "aba_skill_track":
                        skill = row["title"]
                        success = row["message"] == "success"
                        cat = next((c for c, skills in ABA_SKILLS.items() if skill in skills), None)
                        if cat:
                            prog = self.progress[cat][skill]
                            prog.attempts += 1
                            if success:
                                prog.successes += 1
                            prog.last_attempt_ts = time.time() # This is not accurate, but for now it's fine

    def intervene(self, event_category: str, text: Optional[str] = None) -> Optional[BehaviorEvent]:
        """ABA response based on event/behavior."""
        strategies = self.strategy_advisor.suggest(event_category)
        if strategies:
            strategy = random.choice(strategies)
            self.guidance_logger.append(
                BehaviorEvent(
                    timestamp=datetime.utcnow(),
                    level="info",
                    category="aba_strategy_suggestion",
                    title=strategy.title,
                    message=strategy.description,
                    metadata={"event_category": event_category}
                )
            )
            return BehaviorEvent(
                timestamp=datetime.utcnow(),
                level="info",
                category="inner_echo", # to trigger voice output
                title="ABA Strategy",
                message=strategy.description,
            )

        # Tie to skills
        if "anxious" in event_category or "meltdown" in event_category:
            self.generate_social_story("meltdown")
        elif "perseveration" in event_category:
            if text:
                self.redirect_behavior(text)
        elif "encouragement" in event_category:
            if text:
                self.reinforce_success(text)
        return None

    def generate_social_story(self, template_key: str, custom_data: Dict[str, str] | None = None) -> None:
        """Dynamic social story (2025 ABA: Personalized narratives for ToM/social skills)."""
        template = self.social_story_templates.get(template_key, "Let's talk about {topic}.")
        story = template.format(**(custom_data or {}))
        self.guidance_logger.append(
            BehaviorEvent(
                timestamp=datetime.utcnow(),
                level="info",
                category="aba_social_story",
                title=template_key,
                message=story,
            )
        )

    def reinforce_success(self, skill_text: Optional[str] = None) -> None:
        """Positive reinforcement (per ABA ethics: Reward streaks)."""
        skill, cat = self.find_skill(skill_text)
        if skill and cat:
            prog = self.progress[cat][skill]
            prog.successes += 1
            if prog.successes % 3 == 0:  # Streak reward
                reward = random.choice(REWARDS)
                self.guidance_logger.append(
                    BehaviorEvent(
                        timestamp=datetime.utcnow(),
                        level="info",
                        category="inner_echo", # to trigger voice output
                        title="ABA Reinforcement",
                        message=reward,
                        metadata={"skill": skill}
                    )
                )
            if prog.successes / max(prog.attempts, 1) > 0.8:
                prog.current_level = min(prog.current_level + 1, 3)  # Advance level
        self.save_progress()

    def redirect_behavior(self, text: Optional[str]) -> None:
        """Gentle redirection for harmful/repetitive behaviors (2025 ABA: Focus on positive alternatives)."""
        if not text:
            return
        redirection = f"Instead of {text}, let's try {random.choice(['taking a deep breath', 'counting to 10', 'squeezing a fidget toy'])}."
        self.guidance_logger.append(
            BehaviorEvent(
                timestamp=datetime.utcnow(),
                level="info",
                category="inner_echo", # to trigger voice output
                title="ABA Redirection",
                message=redirection,
                metadata={"original_text": text}
            )
        )

    def track_skill_progress(self, skill_text: str, success: bool) -> None:
        """Update progress (integrates with speech_loop attempts)."""
        skill, cat = self.find_skill(skill_text)
        if skill and cat:
            prog = self.progress[cat][skill]
            prog.attempts += 1
            if success:
                prog.successes += 1
            prog.last_attempt_ts = time.time()
            if prog.attempts > 10 and prog.successes / prog.attempts > 0.7:
                self.adapt_skill(skill)  # Phase out old facets
            self.guidance_logger.append(
                BehaviorEvent(
                    timestamp=datetime.utcnow(),
                    level="info",
                    category="aba_skill_track",
                    title=skill,
                    message="success" if success else "failure",
                    metadata={"attempts": str(prog.attempts), "successes": str(prog.successes)}
                )
            )
        self.save_progress()


    def adapt_skill(self, skill: str) -> None:
        """Lifelong adaptation: Phase out old voice facets as skills improve (slow drift)."""
        # Example: Re-sample recent successful attempts
        # This part needs to interact with VoiceProfile. For now, it's a placeholder.
        self.guidance_logger.append(
            BehaviorEvent(
                timestamp=datetime.utcnow(),
                level="info",
                category="aba_adaptation",
                title="Skill Adapted",
                message=f"Adapted strategy for skill: {skill}",
            )
        )


    def find_skill(self, text: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        """Match text to ABA skill."""
        if not text:
            return None, None
        lowered = text.lower()
        for cat, skills in ABA_SKILLS.items():
            for skill in skills:
                if skill in lowered:
                    return skill, cat
        return None, None

    def save_progress(self) -> None:
        """Persist to guidance_events.csv for dashboard."""
        # This will be handled by the guidance logger which already writes to CSV.
        # This method is more for internal state management for the engine.
        pass

    def get_progress_report(self) -> Dict[str, float]:
        """For GUI dashboard: Mastery rates."""
        report = {}
        for cat, skills in self.progress.items():
            for skill, prog in skills.items():
                rate = prog.successes / max(prog.attempts, 1) * 100
                report[f"{cat}:{skill}"] = rate
        return report

