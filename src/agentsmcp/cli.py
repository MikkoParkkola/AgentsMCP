*** Begin Patch ***
*** Update File: src/agentsmcp/cli.py
@@
 @click.group(
     cls=AgentsMCPProgressiveGroup,
     context_settings={"help_option_names": ["-h", "--help"]}
 )
@@
 def main(
     log_level: Optional[str], log_format: Optional[str], config_path: Optional[str], debug: bool,
     network: bool, insecure_mode: bool
 ) -> None:
@@
     main()
*** End Patch ***