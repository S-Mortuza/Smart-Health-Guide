from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os
import re

class ObesityChatbot:
    """
    Hybrid chatbot: Uses fine-tuned GPT-2 + rule-based fallbacks for better responses.
    """

    def __init__(self, model_path: str):
        """Initialize the chatbot"""
        self.use_model = os.path.exists(model_path)
        
        if self.use_model:
            try:
                print(f"Loading fine-tuned model from {model_path}...")
                self.model = GPT2LMHeadModel.from_pretrained(model_path)
                self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model.to(self.device)
                print(f"✓ Model loaded on {self.device}")
            except Exception as e:
                print(f"⚠ Model loading failed: {e}")
                print("  Falling back to rule-based responses")
                self.use_model = False
        else:
            print(f"⚠ Model not found at {model_path}")
            print("  Using rule-based responses")
            self.use_model = False

    def generate_response(self, user_message: str, user_context: dict) -> str:
        """Generate a response using GPT-2 or rules"""
        
        # First try rule-based for common questions (faster and more reliable)
        rule_response = self._try_rule_based(user_message, user_context)
        if rule_response:
            return rule_response
        
        # If no rule match and we have the model, try GPT-2
        if self.use_model:
            try:
                gpt_response = self._generate_gpt2_response(user_message, user_context)
                # Validate response quality
                if self._is_valid_response(gpt_response):
                    return gpt_response
            except Exception as e:
                print(f"GPT-2 generation error: {e}")
        
        # Final fallback
        return self._fallback_response(user_message, user_context)

    def _try_rule_based(self, message: str, context: dict) -> str:
        """Try to match common questions with rule-based responses"""
        msg_lower = message.lower()
        bmi = context.get('bmi', 0)
        prediction = context.get('prediction', 'Unknown')
        faf = context.get('faf', 0)
        fcvc = context.get('fcvc', 0)
        
        # Diet questions
        if any(word in msg_lower for word in ['diet', 'eat', 'food', 'meal', 'nutrition']):
            return (
                f"For {prediction} category, focus on:\n"
                f"• Eat more vegetables and fruits (aim for 5 servings/day)\n"
                f"• Choose lean proteins (fish, chicken, beans)\n"
                f"• Limit processed foods and added sugars\n"
                f"• Control portion sizes\n"
                f"• Drink plenty of water (2-3 liters/day)\n"
                f"Consider consulting a registered dietitian for a personalized meal plan."
            )
        
        # Exercise questions
        if any(word in msg_lower for word in ['exercise', 'workout', 'physical', 'activity', 'gym']):
            advice = f"Your current physical activity: {faf} days/week.\n\n"
            if faf < 2:
                advice += (
                    "Recommendation: Gradually increase to 150 minutes/week of moderate activity:\n"
                    "• Start with 10-15 minute walks daily\n"
                    "• Try swimming, cycling, or dancing\n"
                    "• Take stairs instead of elevators\n"
                    "• Park farther away from destinations"
                )
            else:
                advice += (
                    "Great job staying active! To improve further:\n"
                    "• Add strength training 2-3 times/week\n"
                    "• Gradually increase intensity\n"
                    "• Mix cardio with flexibility exercises\n"
                    "• Stay consistent with your routine"
                )
            return advice
        
        # Weight/BMI questions
        if any(word in msg_lower for word in ['weight', 'lose', 'bmi', 'pounds', 'kg']):
            status = "high" if bmi >= 30 else "elevated" if bmi >= 25 else "normal"
            return (
                f"Your BMI is {bmi:.1f} ({status} range).\n\n"
                f"Healthy weight loss tips:\n"
                f"• Aim for 0.5-1 kg (1-2 lbs) per week\n"
                f"• Combine diet changes with exercise\n"
                f"• Track your food intake\n"
                f"• Get 7-9 hours of sleep\n"
                f"• Manage stress levels\n"
                f"• Stay accountable with a support group\n\n"
                f"Consult your doctor before starting any weight loss program."
            )
        
        # Improvement/change questions
        if any(word in msg_lower for word in ['improve', 'change', 'fix', 'help', 'start', 'begin']):
            return (
                f"To improve from {prediction} category:\n\n"
                f"1. **Nutrition**: Focus on whole foods, vegetables, lean protein\n"
                f"2. **Activity**: {'Increase to 3-5 days/week' if faf < 3 else 'Maintain current level of ' + str(faf) + ' days/week'}\n"
                f"3. **Hydration**: Drink 2-3 liters of water daily\n"
                f"4. **Sleep**: Get 7-9 hours each night\n"
                f"5. **Stress**: Practice meditation or yoga\n"
                f"6. **Tracking**: Monitor your progress weekly\n\n"
                f"Small, consistent changes lead to lasting results!"
            )
        
        # Risk/health questions
        if any(word in msg_lower for word in ['risk', 'health', 'problem', 'danger', 'disease']):
            return (
                f"Category {prediction} may carry increased health risks:\n\n"
                f"• Cardiovascular disease\n"
                f"• Type 2 diabetes\n"
                f"• High blood pressure\n"
                f"• Joint problems\n"
                f"• Sleep apnea\n\n"
                f"However, lifestyle changes can significantly reduce these risks. "
                f"Regular checkups with your doctor are important for monitoring your health."
            )
        
        return None  # No rule matched

    def _generate_gpt2_response(self, message: str, context: dict) -> str:
        """Generate response using fine-tuned GPT-2"""
        prompt = (
            f"Question: {message}\n"
            f"BMI: {context.get('bmi', 'N/A')}, "
            f"Category: {context.get('prediction', 'Unknown')}, "
            f"Activity: {context.get('faf', 'N/A')} days/week\n"
            f"Answer:"
        )
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 80,
                num_return_sequences=1,
                no_repeat_ngram_size=3,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        # Clean response
        sentences = [s.strip() for s in response.split('.') if len(s.strip()) > 15]
        return '. '.join(sentences[:2]) + '.' if sentences else ""

    def _is_valid_response(self, response: str) -> bool:
        """Check if GPT-2 response is good quality"""
        if not response or len(response) < 20:
            return False
        if response.count(' ') < 5:  # Too short
            return False
        # Check for common artifacts
        bad_patterns = ['question:', 'answer:', 'context:', 'weight: 0', 'bmi: 0']
        if any(pattern in response.lower() for pattern in bad_patterns):
            return False
        return True

    def _fallback_response(self, message: str, context: dict) -> str:
        """Generic fallback response"""
        prediction = context.get('prediction', 'Unknown')
        return (
            f"Based on your {prediction} category, I recommend focusing on:\n"
            f"• Balanced, nutritious meals\n"
            f"• Regular physical activity\n"
            f"• Adequate sleep and hydration\n"
            f"• Stress management\n\n"
            f"For specific medical advice, please consult a healthcare professional."
        )