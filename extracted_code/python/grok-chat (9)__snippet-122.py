  2  import React, { useState, useCallback, Suspense, useEffect } from 'reac
     t';
  3 +import * as THREE from 'three';
  4  import type { NodeTarget, ConversationContext, AGISystemState } from '.
     /types';
    ⋮
182            if (!text) return;
182 -          const newTargets = await enhancedService.processConversationR
     esponse(text, false);
183 +          // Fast client-side mapping from words to 3D nodes for latenc
     y-free visuals.
184 +          const newTargets = makeTargetsFromText(text);
185            setTargets(newTargets);
184 -          setContext(enhancedService.getCurrentContext());
186 +          setContext({ imagePrompt: text });
187          } catch (err) {

