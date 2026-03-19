#!/usr/bin/env bun
import { SkillCache } from './src/skill-cache.ts';

async function main() {
  const args = Bun.argv.slice(2);
  if (args.length < 2) {
    console.error('Usage: reminder <tool_name> <json_args>');
    process.exit(1);
  }

  const toolName = args[0];
  const jsonArgs = JSON.parse(args[1]);

  try {
    const cache = new SkillCache();
    let result: any;
    if (toolName === 'top_skills') {
      const skills = await cache.topSkillsForPrompt(
        jsonArgs.prompt,
        jsonArgs.topK || 3,
      );
      result = skills;
    } else {
      console.error(`Unknown tool: ${toolName}`);
      process.exit(1);
    }
    process.stdout.write(JSON.stringify(result));
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  }
}

main();
